# infer_glmocr.py - GLM-OCR, LMDB -> JSONL
import os
import json
import lmdb
import cv2
import numpy as np
import unicodedata
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import torch
from modelscope import AutoProcessor, AutoModelForImageTextToText


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
    try:
        img = Image.open(BytesIO(image_bin)).convert("RGB")
        return img
    except Exception:
        return None


# -------------------- Model loading --------------------

def load_model(model_path="ZhipuAI/GLM-OCR"):
    print(f"Loading {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, processor


# -------------------- Single LMDB inference --------------------

def dump_one_lmdb_to_jsonl(
    lmdb_path: str,
    model,
    processor,
    out_jsonl: str,
    prompt: str = "Text Recognition:",
):
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    num_samples = _read_num_samples(env)
    dataset_name = os.path.basename(os.path.normpath(lmdb_path))

    skipped = 0

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        with env.begin(write=False) as txn:
            for idx in tqdm(range(1, num_samples + 1), desc=f"GLM-OCR | {dataset_name}"):
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

                # Build messages in GLM-OCR format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                try:
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(model.device)
                    inputs.pop("token_type_ids", None)

                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=8192)

                    recognized_text = processor.decode(
                        generated_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    recognized_text = recognized_text.replace("\n", " ").replace("\r", " ").strip()
                except Exception as e:
                    recognized_text = ""

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

    model_path = "ZhipuAI/GLM-OCR"
    out_dir = "./glmocr_results"
    os.makedirs(out_dir, exist_ok=True)

    model, processor = load_model(model_path)

    for p in lmdb_list:
        ds = os.path.basename(os.path.normpath(p))
        out_jsonl = os.path.join(out_dir, f"{ds}.jsonl")
        summary = dump_one_lmdb_to_jsonl(p, model, processor, out_jsonl)
        print("[DUMP]", summary)
