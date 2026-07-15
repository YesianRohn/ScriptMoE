"""End-to-end evaluation of the E2EOCR pipeline on the CC-OCR multi_lan_ocr
benchmark.

This script mirrors the workflow of
``OpenOCR-MOE/tools/run_cc_ocr_multi_lan.py`` but drives the self-contained
``E2EOCR`` pipeline (``E2EOCR/infer.py``) instead, so the text **detector is
selectable** and both the **detection / recognition score thresholds** are
configurable.

Pipeline
--------
1. Read the local TSV bundles from ``CC-OCR-MLT/tsv/<Lang>_150.tsv``.
   Columns: index, image (base64 jpeg), image_name, question, answer,
   category, l2-category, split.
2. For every row: decode the base64 image, run the E2EOCR pipeline
   (``--det`` selects openocr / ppv5 / ppv6 / rec_only), and keep every
   box together with its detection score and recognition score.
3. Filter boxes by independent thresholds (``--det_thresh`` / ``--rec_thresh``)
   and concatenate the surviving transcriptions (top-to-bottom, left-to-right)
   into a single response string.
4. Write, under ``--work_dir`` (default ``cc_ocr_eval``):
     data_root/index/multi_lan_ocr.json           # evaluation index
     data_root/data/multi_lan_ocr/<Lang>/<Lang>/label.json   # GT
     results/<exp_name>/<Lang>/<image_name>.json             # prediction
     results/<exp_name>/<Lang>/det_rec/<image_name>.json     # raw per-box
5. Optionally launch the official evaluator
   (``CC-OCR-MLT/eval/main.py``) unless ``--no_eval`` is given.

The prediction / GT layout is byte-for-byte compatible with the vendored
evaluator: the evaluator strips a single ``.json`` extension from the
prediction file name, so keeping ``<image_name>.json`` (image_name already
contains ``.jpg``) makes the response key equal the GT key.

Examples
--------
    # default: openocr detector, det/rec thresh 0.5, all languages
    python run_cc_ocr_mlt.py

    # PP-OCRv5 detector, custom thresholds, only a few languages
    python run_cc_ocr_mlt.py --det ppv5 --det_thresh 0.3 --rec_thresh 0.6 \
        --languages French German --max_per_lang 50

    # recognise the whole image only (no detection)
    python run_cc_ocr_mlt.py --det rec_only --exp_name rec_only_run
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
E2EOCR_DIR = ROOT / "E2EOCR"
sys.path.insert(0, str(E2EOCR_DIR))

from infer import EndToEndOCR, DEFAULT_WEIGHTS, DEFAULT_CHAR_DICT  # noqa: E402

LANGUAGES = [
    "Arabic", "French", "German", "Italian", "Japanese",
    "Korean", "Portuguese", "Russian", "Spanish", "Vietnamese",
]

DEFAULT_TSV_DIR = ROOT / "CC-OCR-MLT" / "tsv"
DEFAULT_EVAL_MAIN = ROOT / "CC-OCR-MLT" / "eval" / "main.py"


# ---------------------------------------------------------------------------
def iter_tsv(tsv_path: Path):
    """Yield (image_name, answer, image_bgr) for each valid TSV row."""
    csv.field_size_limit(sys.maxsize)
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            img_b64 = row.get("image", "")
            image_name = row.get("image_name", "").strip()
            answer = row.get("answer", "")
            if not img_b64 or not image_name:
                continue
            try:
                raw = base64.b64decode(img_b64)
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[warn] failed to decode {image_name}: {e}")
                continue
            yield image_name, answer, img_bgr


# ---------------------------------------------------------------------------
def filter_results(res, det_thresh, rec_thresh):
    """Keep boxes whose detection AND recognition scores both pass."""
    kept = []
    for r in res:
        det_score = float(r.get("det_score", 1.0))
        rec_score = float(r.get("score", 1.0))
        if det_score >= det_thresh and rec_score >= rec_thresh:
            kept.append(r)
    return kept


def boxes_top_to_bottom(res):
    """Sort lines top-to-bottom, left-to-right; join into a single string."""
    if not res:
        return ""
    items = []
    for r in res:
        pts = np.array(r["points"], dtype=np.float32)
        items.append((float(pts[:, 1].min()), float(pts[:, 0].min()),
                      r["transcription"]))
    items.sort(key=lambda x: (x[0], x[1]))
    return " ".join(t for _, _, t in items)


def build_det_rec_records(res):
    """Serialise raw per-box detection + recognition results."""
    return [{
        "points": r["points"],
        "transcription": r["transcription"],
        "det_score": round(float(r.get("det_score", 1.0)), 6),
        "rec_score": round(float(r.get("score", 1.0)), 6),
    } for r in res]


# ---------------------------------------------------------------------------
def write_index(index_path: Path, languages):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{
        "dataset": lang,
        "base_dir": f"data/multi_lan_ocr/{lang}/{lang}",
        "group": "multi_lan_ocr",
        "op": lang,
        "num": 150,
    } for lang in languages]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return index_path


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate the E2EOCR pipeline on CC-OCR multi_lan_ocr.")
    # --- data / output layout ---
    ap.add_argument("--tsv_dir", type=str, default=str(DEFAULT_TSV_DIR),
                    help="Directory holding <Lang>_150.tsv bundles.")
    ap.add_argument("--work_dir", type=str, default=str(ROOT / "cc_ocr_eval"),
                    help="Root for the generated data_root / results tree.")
    ap.add_argument("--exp_name", type=str, default=None,
                    help="Result sub-dir name (default derives from det+thresh).")
    ap.add_argument("--languages", nargs="+", default=LANGUAGES,
                    help="Subset of languages to evaluate.")
    ap.add_argument("--max_per_lang", type=int, default=150,
                    help="Limit images per language (debug).")
    ap.add_argument("--skip_done", action="store_true", default=True,
                    help="Skip images whose prediction already exists.")
    ap.add_argument("--no_eval", action="store_true", default=False,
                    help="Only run inference; do not launch the evaluator.")
    ap.add_argument("--eval_main", type=str, default=str(DEFAULT_EVAL_MAIN),
                    help="Path to CC-OCR-MLT/eval/main.py.")

    # --- detector selection ---
    ap.add_argument("--det", default="openocr",
                    choices=["openocr", "ppv5", "ppv6", "rec_only"],
                    help="Text detector (default: openocr). "
                         "rec_only feeds the whole image to the recogniser.")
    ap.add_argument("--det_box_type", default="quad",
                    choices=["quad", "poly"],
                    help="Crop mode for detected boxes.")

    # --- thresholds ---
    ap.add_argument("--det_thresh", type=float, default=0,
                    help="Detection score threshold (filter boxes).")
    ap.add_argument("--rec_thresh", type=float, default=0.7,
                    help="Recognition score threshold (filter boxes).")
    ap.add_argument("--drop_score", type=float, default=0.0,
                    help="Recognition score below which a box is dropped "
                         "inside the pipeline BEFORE thresholding. Keep 0.0 so "
                         "--det_thresh / --rec_thresh fully control filtering.")

    # --- recogniser ---
    ap.add_argument("--rec_batch_num", type=int, default=8)
    ap.add_argument("--max_text_length", type=int, default=100)
    ap.add_argument("--max_ratio", type=int, default=20)
    ap.add_argument("--use_gpu", default="auto",
                    choices=["auto", "true", "false"])

    # --- misc ---
    ap.add_argument("--max_side", type=int, default=2000,
                    help="Downscale so max(H,W) <= this before detection. "
                         "0 disables.")
    ap.add_argument("--show_pred", action="store_true", default=False,
                    help="Also print the predicted text in progress logs. "
                         "Off by default: only score results are shown.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    for lang in args.languages:
        if lang not in LANGUAGES:
            print(f"[warn] unknown language ignored: {lang}")
    languages = [l for l in args.languages if l in LANGUAGES]
    if not languages:
        print("[error] no valid languages to evaluate.")
        return

    exp_name = args.exp_name or (
        f"{args.det}_d{args.det_thresh:g}_r{args.rec_thresh:g}")

    work_dir = Path(args.work_dir)
    data_root = work_dir / "data_root"
    gt_root = data_root / "data" / "multi_lan_ocr"
    index_path = data_root / "index" / "multi_lan_ocr.json"
    out_root = work_dir / "results" / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    write_index(index_path, languages)

    tsv_dir = Path(args.tsv_dir)

    print(f"[init] det={args.det} mode=server "
          f"det_thresh={args.det_thresh} rec_thresh={args.rec_thresh} "
          f"exp={exp_name}")
    pipeline = EndToEndOCR(
        det=args.det,
        weights_path=DEFAULT_WEIGHTS,
        character_dict_path=DEFAULT_CHAR_DICT,
        det_model_dir=None,
        lang="ch",
        drop_score=args.drop_score,
        det_box_type=args.det_box_type,
        use_gpu=args.use_gpu,
        rec_batch_num=args.rec_batch_num,
        max_text_length=args.max_text_length,
        max_ratio=args.max_ratio,
        det_mode="server",
    )

    overall_t0 = time.time()
    for lang in languages:
        tsv_path = tsv_dir / f"{lang}_150.tsv"
        if not tsv_path.exists():
            print(f"[warn] TSV not found, skip {lang}: {tsv_path}")
            continue

        lang_gt_dir = gt_root / lang / lang
        lang_gt_dir.mkdir(parents=True, exist_ok=True)
        gt_label_path = lang_gt_dir / "label.json"
        gt_label = {}

        lang_pred_dir = out_root / lang
        lang_pred_dir.mkdir(parents=True, exist_ok=True)
        lang_detrec_dir = lang_pred_dir / "det_rec"
        lang_detrec_dir.mkdir(parents=True, exist_ok=True)

        n_done, t_lang = 0, time.time()
        for image_name, answer, img_bgr in iter_tsv(tsv_path):
            if n_done >= args.max_per_lang:
                break
            gt_label[image_name] = answer

            pred_path = lang_pred_dir / f"{image_name}.json"
            detrec_path = lang_detrec_dir / f"{image_name}.json"
            if args.skip_done and pred_path.exists() and detrec_path.exists():
                n_done += 1
                continue

            t0 = time.time()
            try:
                if args.max_side > 0:
                    h, w = img_bgr.shape[:2]
                    m = max(h, w)
                    if m > args.max_side:
                        s = args.max_side / float(m)
                        img_bgr = cv2.resize(
                            img_bgr,
                            (int(round(w * s)), int(round(h * s))),
                            interpolation=cv2.INTER_AREA)
                res, _ = pipeline.infer(img_bgr)
            except Exception as e:
                print(f"[warn] OCR failed on {lang}/{image_name}: {e}")
                res = []

            kept = filter_results(res, args.det_thresh, args.rec_thresh)
            response_text = boxes_top_to_bottom(kept)
            elapsed = time.time() - t0

            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump({
                    "image": f"{lang}/images/{image_name}",
                    "model_name": "local_scriptmoe",
                    "response": response_text,
                }, f, ensure_ascii=False)

            with open(detrec_path, "w", encoding="utf-8") as f:
                json.dump({
                    "image": f"{lang}/images/{image_name}",
                    "model_name": "local_scriptmoe",
                    "det": args.det,
                    "det_thresh": args.det_thresh,
                    "rec_thresh": args.rec_thresh,
                    "num_boxes": len(res),
                    "num_kept": len(kept),
                    "results": build_det_rec_records(res),
                }, f, ensure_ascii=False, indent=2)

            n_done += 1
            if n_done == 1 or n_done % 10 == 0:
                msg = (f"[{lang}] {n_done}/{args.max_per_lang}  "
                       f"t={elapsed:.2f}s boxes={len(res)} kept={len(kept)}")
                if args.show_pred:
                    msg += f"  pred='{response_text[:50]}'"
                print(msg)

        with open(gt_label_path, "w", encoding="utf-8") as f:
            json.dump(gt_label, f, ensure_ascii=False, indent=2)
        print(f"== {lang} done: {n_done} samples, "
              f"took {time.time()-t_lang:.1f}s  GT={gt_label_path}")

    print(f"[inference] all languages finished in "
          f"{time.time()-overall_t0:.1f}s")

    # ---- evaluation ----
    if args.no_eval:
        print("[eval] skipped (--no_eval). Run manually:\n"
              f"  python {args.eval_main} {index_path} {out_root}")
        return

    eval_main = Path(args.eval_main)
    if not eval_main.exists():
        print(f"[eval] evaluator not found: {eval_main}")
        return
    print(f"[eval] launching evaluator: {eval_main}")
    ret = subprocess.call(
        [sys.executable, str(eval_main), str(index_path), str(out_root)])
    if ret == 0:
        print(f"[eval] done. summary.md at: {out_root.parent / 'summary.md'}")
    else:
        print(f"[eval] evaluator exited with code {ret}. You can rerun:\n"
              f"  python {eval_main} {index_path} {out_root}")


if __name__ == "__main__":
    main()
