"""End-to-end multilingual OCR command-line tool.

Pipeline = optional text detector + ScriptMoE recogniser.

Detector is selectable via ``--det``:

* ``openocr`` (default) : OpenOCR text detector (``openocr-python`` package).
* ``ppv5``             : PaddleOCR PP-OCRv5 server detector.
* ``ppv6``             : PaddleOCR PP-OCRv6 medium detector.
* ``rec_only``         : no detection; the whole image is fed to the recogniser.

Input  : a single image file, or a directory of images.
Output : results printed to the terminal + a JSON file.

This folder is fully self-contained: it only relies on the sibling modules
``modeling.py`` / ``postprocess.py`` / ``visualize.py`` plus a few public PyPI
packages. The recogniser weights (``model.safetensors``) and the character
dictionary (``assets/dict.txt``) live next to this script.

Examples
--------
    # OpenOCR detector (default)
    python infer.py --image path/to/img.jpg

    # PP-OCRv5 detector, write JSON next to the image
    python infer.py --image path/to/img.jpg --det ppv5

    # recognise the whole image only (no detection)
    python infer.py --image path/to/crop.jpg --det rec_only

    # batch a directory
    python infer.py --image path/to/dir --output out.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

from modeling import ScriptMoERecModel
from postprocess import ARLabelDecoder
from visualize import get_minarea_rect_crop, get_rotate_crop_image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'

DEFAULT_CHAR_DICT = str(ASSETS / 'dict.txt')
DEFAULT_WEIGHTS = str(ROOT / 'model.safetensors')

# Unified recogniser settings (same across every detector / rec_only).
DEFAULT_MAX_TEXT_LENGTH = 100
DEFAULT_MAX_RATIO = 20

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


# =============================================================================
# Detectors
# =============================================================================
class _OpenOCRDetector:
    """Thin wrapper around the ``openocr-python`` text detector (task='det')."""

    def __init__(self, use_gpu: str = 'auto', mode: str = 'mobile'):
        try:
            from openocr import OpenOCR
        except ImportError as e:
            raise ImportError(
                'openocr is required for the openocr detector. '
                'pip install openocr-python') from e
        self.detector = OpenOCR(task='det', mode=mode, use_gpu=use_gpu)

    def __call__(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import tempfile

        fd, tmp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        try:
            cv2.imwrite(tmp_path, img_bgr)
            results = self.detector(image_path=tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not results:
            return _empty_det()
        res = results[0]
        if not isinstance(res, dict):
            return _empty_det()
        polys = res.get('boxes')
        scores = None
        for k in ('scores', 'det_scores', 'box_scores'):
            if res.get(k) is not None:
                scores = res[k]
                break
        return _to_quads_scores(polys, scores)


class _PaddleOCRDetector:
    """Thin wrapper around a PaddleOCR detector (PP-OCRv5 / PP-OCRv6)."""

    def __init__(self, ocr_version: str, model_name: str,
                 use_gpu: str = 'auto', det_model_dir: str = None,
                 lang: str = 'ch'):
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                'paddleocr is required for the ppv5/ppv6 detector. '
                'pip install paddleocr paddlepaddle') from e

        if use_gpu == 'auto':
            try:
                import paddle
                use_gpu_flag = paddle.device.is_compiled_with_cuda()
            except Exception:
                use_gpu_flag = False
        elif use_gpu == 'true':
            use_gpu_flag = True
        else:
            use_gpu_flag = False

        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            ocr_version=ocr_version,
            text_detection_model_name=model_name,
            lang=lang,
            device='gpu' if use_gpu_flag else 'cpu',
        )
        if det_model_dir is not None:
            kwargs['text_detection_model_dir'] = det_model_dir
        self.engine = PaddleOCR(**kwargs)

    def __call__(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        results = self.engine.predict(img_bgr)
        if not results:
            return _empty_det()

        res = results[0]
        data = None
        if hasattr(res, 'json'):
            data = res.json
            if isinstance(data, dict) and 'res' in data:
                data = data['res']
        elif isinstance(res, dict):
            data = res

        polys, scores = None, None
        if isinstance(data, dict):
            for k in ('dt_polys', 'rec_polys', 'polys', 'boxes'):
                if data.get(k) is not None:
                    polys = data[k]
                    break
            for k in ('dt_scores', 'rec_scores', 'scores'):
                if data.get(k) is not None:
                    scores = data[k]
                    break
        return _to_quads_scores(polys, scores)


def _empty_det() -> Tuple[np.ndarray, np.ndarray]:
    return (np.zeros((0, 4, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32))


def _to_quads_scores(polys, scores) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise detector output to ``(N, 4, 2)`` quads + ``(N,)`` scores.

    Missing / mismatched scores fall back to neutral ``1.0`` confidences.
    """
    if polys is None or len(polys) == 0:
        return _empty_det()
    polys = np.array(polys, dtype=np.float32)
    if polys.ndim == 3 and polys.shape[1] != 4:
        quads = []
        for p in polys:
            rect = cv2.minAreaRect(p.astype(np.int32))
            quads.append(cv2.boxPoints(rect))
        polys = np.array(quads, dtype=np.float32)

    n = len(polys)
    if scores is None:
        scores = np.ones((n,), dtype=np.float32)
    else:
        scores = np.array(scores, dtype=np.float32).reshape(-1)
        if len(scores) != n:
            scores = np.ones((n,), dtype=np.float32)
    return polys, scores


def build_detector(det: str, use_gpu: str = 'auto', det_model_dir: str = None,
                   lang: str = 'ch', det_mode: str = 'mobile'):
    """Return a detector callable, or ``None`` for ``rec_only``."""
    if det == 'rec_only':
        return None
    if det == 'openocr':
        return _OpenOCRDetector(use_gpu=use_gpu, mode=det_mode)
    if det == 'ppv5':
        return _PaddleOCRDetector('PP-OCRv5', 'PP-OCRv5_server_det',
                                  use_gpu=use_gpu, det_model_dir=det_model_dir,
                                  lang=lang)
    if det == 'ppv6':
        return _PaddleOCRDetector('PP-OCRv6', 'PP-OCRv6_medium_det',
                                  use_gpu=use_gpu, det_model_dir=det_model_dir,
                                  lang=lang)
    raise ValueError(f'unknown detector: {det}')


# =============================================================================
# Recogniser: ScriptMoE (eval-aligned preprocessing)
# =============================================================================
def _load_state_dict(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'recogniser weights not found: {path}')
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    return obj


class _ScriptMoERecognizer:
    """ScriptMoE recogniser — preprocessing mirrors ``RatioDataSetTVResize``.

    1. If h > 1.5·w, rotate vertical text 90° → horizontal.
    2. ``gen_ratio = clip(round(w/h), min_ratio, max_ratio)``.
    3. Target shape is ``base_shape[gen_ratio-1]`` for ratio<=4, else
       ``(base_h*gen_ratio, base_h)``.
    4. Resize (BICUBIC) → ToTensor → Normalize(0.5, 0.5).
    """

    BASE_SHAPE = [(64, 64), (96, 48), (112, 40), (128, 32)]
    BASE_H = 32
    MIN_RATIO = 1

    def __init__(self, weights_path: str, character_dict_path: str,
                 use_gpu: str = 'auto',
                 max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
                 max_ratio: int = DEFAULT_MAX_RATIO):
        if use_gpu == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif use_gpu == 'true':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.max_ratio = int(max_ratio)
        self.decoder = ARLabelDecoder(character_dict_path, use_space_char=True)
        self.num_classes = self.decoder.num_classes

        self.model = ScriptMoERecModel(num_classes=self.num_classes,
                                       max_len=max_text_length)
        state = _load_state_dict(weights_path)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f'[recognizer] missing keys: {len(missing)} '
                  f'(first 3: {missing[:3]})')
        if unexpected:
            print(f'[recognizer] unexpected keys: {len(unexpected)} '
                  f'(first 3: {unexpected[:3]})')
        self.model.to(self.device).eval()

        self._to_tensor = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    def _preprocess(self, pil_img: Image.Image) -> Tuple[torch.Tensor, int]:
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        w, h = pil_img.size
        if h > w * 1.5:
            pil_img = pil_img.transpose(Image.ROTATE_90)
            w, h = pil_img.size

        gen_ratio = max(1, round(float(w) / float(h)))
        gen_ratio = int(np.clip(gen_ratio, self.MIN_RATIO, self.max_ratio))
        if gen_ratio <= 4:
            img_w, img_h = self.BASE_SHAPE[gen_ratio - 1]
        else:
            img_w, img_h = self.BASE_H * gen_ratio, self.BASE_H

        resized = F.resize(pil_img, (img_h, img_w),
                           interpolation=T.InterpolationMode.BICUBIC)
        return self._to_tensor(resized), gen_ratio

    @torch.no_grad()
    def __call__(self, pil_imgs: List[Image.Image],
                 batch_num: int = 8) -> List[dict]:
        n = len(pil_imgs)
        if n == 0:
            return []

        tensors, ratios = [], []
        for img in pil_imgs:
            t, r = self._preprocess(img)
            tensors.append(t)
            ratios.append(r)

        buckets: dict = {}
        for i, r in enumerate(ratios):
            buckets.setdefault(r, []).append(i)

        results: List[dict] = [None] * n
        for _, idx_list in buckets.items():
            for s in range(0, len(idx_list), batch_num):
                sub = idx_list[s:s + batch_num]
                batch = torch.stack([tensors[i] for i in sub], dim=0).to(
                    self.device)
                logits = self.model(batch)
                decoded = self.decoder(logits)
                for k, (text, score) in zip(sub, decoded):
                    results[k] = {'text': text, 'score': float(score)}
        return results


# =============================================================================
# End-to-end pipeline
# =============================================================================
class EndToEndOCR:
    """Optional detector + ScriptMoE recogniser.

    ``infer`` returns ``(results, timing)`` where each result is a dict with
    ``transcription`` / ``points`` / ``score``.
    """

    def __init__(
        self,
        det: str = 'openocr',
        weights_path: str = DEFAULT_WEIGHTS,
        character_dict_path: str = DEFAULT_CHAR_DICT,
        det_model_dir: str = None,
        lang: str = 'ch',
        drop_score: float = 0.5,
        det_box_type: str = 'quad',
        use_gpu: str = 'auto',
        rec_batch_num: int = 8,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        max_ratio: int = DEFAULT_MAX_RATIO,
        det_mode: str = 'mobile',
    ):
        self.det = det
        self.drop_score = drop_score
        self.det_box_type = det_box_type
        self.rec_batch_num = rec_batch_num
        self.detector = build_detector(det, use_gpu=use_gpu,
                                        det_model_dir=det_model_dir,
                                        lang=lang, det_mode=det_mode)
        self.recognizer = _ScriptMoERecognizer(
            weights_path=weights_path,
            character_dict_path=character_dict_path,
            use_gpu=use_gpu,
            max_text_length=max_text_length,
            max_ratio=max_ratio,
        )

    def _crop_boxes(self, img_bgr, boxes):
        crops = []
        for box in boxes:
            box = np.array(copy.deepcopy(box)).astype(np.float32)
            if self.det_box_type == 'quad':
                crops.append(get_rotate_crop_image(img_bgr, box))
            else:
                crops.append(get_minarea_rect_crop(img_bgr, box))
        return crops

    @staticmethod
    def _sort_boxes_scores(boxes: np.ndarray, scores: np.ndarray):
        """Sort boxes top-to-bottom, left-to-right, keeping scores aligned."""
        n = len(boxes)
        if n == 0:
            return [], []
        idxs = sorted(range(n), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
        for i in range(n - 1):
            for j in range(i, -1, -1):
                bj, bj1 = boxes[idxs[j]], boxes[idxs[j + 1]]
                if abs(bj1[0][1] - bj[0][1]) < 10 and bj1[0][0] < bj[0][0]:
                    idxs[j], idxs[j + 1] = idxs[j + 1], idxs[j]
                else:
                    break
        return [boxes[i] for i in idxs], [float(scores[i]) for i in idxs]

    def infer(self, img_bgr: np.ndarray, drop_score: float = None,
              rec_batch_num: int = None):
        if drop_score is None:
            drop_score = self.drop_score
        if rec_batch_num is None:
            rec_batch_num = self.rec_batch_num

        ori = img_bgr.copy()
        h, w = ori.shape[:2]

        # ---- rec_only: feed the whole image straight to the recogniser ----
        if self.detector is None:
            t1 = time.time()
            pil_img = Image.fromarray(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
            rec = self.recognizer([pil_img], batch_num=rec_batch_num)
            rec_time = time.time() - t1
            r = rec[0]
            results = []
            if r['score'] >= drop_score:
                results.append({
                    'transcription': r['text'],
                    'points': [[0, 0], [w, 0], [w, h], [0, h]],
                    'score': float(r['score']),
                    'det_score': 1.0,
                })
            return results, {
                'detection_time': 0.0,
                'recognition_time': rec_time,
                'time_cost': rec_time,
                'num_boxes': 1,
            }

        # ---- detect -> crop -> recognise ----
        t0 = time.time()
        boxes, det_scores = self.detector(ori)
        det_time = time.time() - t0

        if len(boxes) == 0:
            return [], {'detection_time': det_time, 'recognition_time': 0.0,
                        'time_cost': det_time, 'num_boxes': 0}

        boxes, det_scores = self._sort_boxes_scores(
            np.array(boxes), np.array(det_scores))
        crops = self._crop_boxes(ori, boxes)

        t1 = time.time()
        pil_crops = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                     for c in crops]
        rec = self.recognizer(pil_crops, batch_num=rec_batch_num)
        rec_time = time.time() - t1

        results = []
        for box, det_score, r in zip(boxes, det_scores, rec):
            if r['score'] >= drop_score:
                results.append({
                    'transcription': r['text'],
                    'points': np.array(box).tolist(),
                    'score': float(r['score']),
                    'det_score': float(det_score),
                })
        return results, {
            'detection_time': det_time,
            'recognition_time': rec_time,
            'time_cost': det_time + rec_time,
            'num_boxes': int(len(boxes)),
        }


# =============================================================================
# CLI
# =============================================================================
def _collect_images(path: str) -> List[str]:
    p = Path(path)
    if p.is_dir():
        return sorted(str(f) for f in p.iterdir()
                      if f.suffix.lower() in IMAGE_EXTS)
    if p.is_file():
        return [str(p)]
    raise FileNotFoundError(f'image path not found: {path}')


def _read_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        # Fallback for paths / formats OpenCV cannot read directly.
        try:
            pil = Image.open(path).convert('RGB')
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def parse_args():
    ap = argparse.ArgumentParser(
        description='End-to-end multilingual OCR (detector + ScriptMoE rec).')
    ap.add_argument('--image', required=True,
                    help='Path to an image file or a directory of images.')
    ap.add_argument('--det', default='openocr',
                    choices=['openocr', 'ppv5', 'ppv6', 'rec_only'],
                    help='Text detector (default: openocr). '
                         'rec_only = recognise the whole image.')
    ap.add_argument('--output', default=None,
                    help='Output JSON path (default: ocr_result.json in cwd).')
    ap.add_argument('--drop_score', type=float, default=0.5,
                    help='Discard predictions below this confidence.')
    ap.add_argument('--rec_batch_num', type=int, default=8,
                    help='Recogniser batch size.')
    ap.add_argument('--use_gpu', default='auto',
                    choices=['auto', 'true', 'false'],
                    help='Device selection (default: auto).')
    ap.add_argument('--det_box_type', default='quad',
                    choices=['quad', 'poly'],
                    help='Crop mode for detected boxes.')
    ap.add_argument('--max_text_length', type=int,
                    default=DEFAULT_MAX_TEXT_LENGTH,
                    help='Recogniser max decode length (default: 100).')
    ap.add_argument('--max_ratio', type=int, default=DEFAULT_MAX_RATIO,
                    help='Recogniser max width/height ratio (default: 20).')
    return ap.parse_args()


def main():
    args = parse_args()
    images = _collect_images(args.image)
    if not images:
        print(f'[error] no images found under: {args.image}')
        return

    print(f'[init] det={args.det} | images={len(images)} | '
          f'weights={DEFAULT_WEIGHTS}')
    pipeline = EndToEndOCR(
        det=args.det,
        weights_path=DEFAULT_WEIGHTS,
        character_dict_path=DEFAULT_CHAR_DICT,
        det_model_dir=None,
        lang='ch',
        drop_score=args.drop_score,
        det_box_type=args.det_box_type,
        use_gpu=args.use_gpu,
        rec_batch_num=args.rec_batch_num,
        max_text_length=args.max_text_length,
        max_ratio=args.max_ratio,
        det_mode='server',
    )

    all_output = []
    for img_path in images:
        img_bgr = _read_image(img_path)
        if img_bgr is None:
            print(f'[warn] cannot read image, skipped: {img_path}')
            continue

        results, timing = pipeline.infer(img_bgr)

        print(f'\n=== {img_path} ===')
        print(f'det={timing["detection_time"]*1000:.1f}ms  '
              f'rec={timing["recognition_time"]*1000:.1f}ms  '
              f'total={timing["time_cost"]*1000:.1f}ms  '
              f'boxes={timing["num_boxes"]}  kept={len(results)}')
        for i, r in enumerate(results):
            print(f'  [{i}] ({r["score"]:.3f}) {r["transcription"]}')

        all_output.append({
            'image': img_path,
            'det': args.det,
            'results': results,
            'timing': timing,
        })

    out_path = args.output or os.path.join(os.getcwd(), 'ocr_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
    print(f'\n[done] results written to: {out_path}')


if __name__ == '__main__':
    main()
