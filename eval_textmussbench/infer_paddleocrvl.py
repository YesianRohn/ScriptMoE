# infer_paddleocrvl.py - PaddleOCR-VL-1.5, LMDB -> JSONL
import os
import json
import lmdb
import cv2
import numpy as np
import unicodedata
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# -------------------- LMDB reading utilities --------------------

def _read_num_samples(env: lmdb.Environment) -> int:
    with env.begin(write=False) as txn:
        v = txn.get(b"num-samples")
        if v is None:
            raise RuntimeError("LMDB missing key: num-samples")
        return int(v.decode("utf-8"))


def _get_item(txn, idx: int):
    image_key = f"image-{idx:09d}".encode()
    label_key = f"label-{idx:09d}".encode()
    image_bin = txn.get(image_key)
    label_bin = txn.get(label_key)
    if image_bin is None or label_bin is None:
        return None, None
    try:
        label = label_bin.decode("utf-8")
    except Exception:
        label = label_bin.decode("utf-8", errors="ignore")
    label = unicodedata.normalize("NFKC", label)
    return image_bin, label


def _decode_image_bin_to_pil(image_bin: bytes):
    arr = np.frombuffer(image_bin, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# -------------------- Model loading --------------------

def load_model(model_path="PaddlePaddle/PaddleOCR-VL-1.5"):
    print(f"Loading {model_path} ...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# -------------------- Single LMDB inference --------------------

def dump_one_lmdb_to_jsonl(
    lmdb_path: str,
    model,
    processor,
    out_jsonl: str,
    batch_size: int = 1,
    task: str = "ocr",
):
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    num_samples = _read_num_samples(env)
    dataset_name = os.path.basename(os.path.normpath(lmdb_path))

    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
        "spotting": "Spotting:",
        "seal": "Seal Recognition:",
    }
    prompt_text = PROMPTS.get(task, "OCR:")
    max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28

    skipped = 0

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        with env.begin(write=False) as txn:
            for idx in tqdm(range(1, num_samples + 1), desc=f"PaddleOCR-VL | {dataset_name}"):
                image_bin, label = _get_item(txn, idx)
                if image_bin is None:
                    skipped += 1
                    fw.write(json.dumps({
                        "dataset": dataset_name, "lmdb_path": lmdb_path,
                        "idx": idx, "gt": label if label else "",
                        "pred": "", "pred_score": None,
                        "skipped": True, "skip_reason": "missing_image_or_label",
                    }, ensure_ascii=False) + "\n")
                    continue

                pil_img = _decode_image_bin_to_pil(image_bin)
                if pil_img is None:
                    skipped += 1
                    fw.write(json.dumps({
                        "dataset": dataset_name, "lmdb_path": lmdb_path,
                        "idx": idx, "gt": label,
                        "pred": "", "pred_score": None,
                        "skipped": True, "skip_reason": "decode_failed",
                    }, ensure_ascii=False) + "\n")
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    images_kwargs={
                        "size": {
                            "shortest_edge": processor.image_processor.min_pixels,
                            "longest_edge": max_pixels,
                        }
                    },
                ).to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=512)
                result = processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:-1]
                )
                recognized_text = result.replace("\n", " ").replace("\r", " ").strip()

                row = {
                    "dataset": dataset_name,
                    "lmdb_path": lmdb_path,
                    "idx": idx,
                    "gt": label,
                    "pred": recognized_text,
                    "pred_score": None,
                    "skipped": False,
                }
                fw.write(json.dumps(row, ensure_ascii=False) + "\n")

    env.close()
    return {"lmdb_path": lmdb_path, "dataset": dataset_name,
            "num_samples": num_samples, "skipped": skipped}


# -------------------- main --------------------

if __name__ == "__main__":
    lmdb_list = [

    ]

    model_path = "PaddlePaddle/PaddleOCR-VL-1.5"
    out_dir = "./paddleocrvl_results"
    os.makedirs(out_dir, exist_ok=True)

    model, processor = load_model(model_path)

    for p in lmdb_list:
        ds = os.path.basename(os.path.normpath(p))
        out_jsonl = os.path.join(out_dir, f"{ds}.jsonl")
        summary = dump_one_lmdb_to_jsonl(p, model, processor, out_jsonl)
        print("[DUMP]", summary)
