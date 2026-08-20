# infer_hunyuanocr.py - Tencent HunyuanOCR, LMDB -> JSONL
import os
import json
import lmdb
import cv2
import numpy as np
import unicodedata
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


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
    """Decode image bytes to a PIL RGB image."""
    arr = np.frombuffer(image_bin, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# -------------------- Model loading --------------------

def load_model(model_name="tencent/HunyuanOCR"):
    print(f"Loading {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor


# -------------------- Single LMDB inference --------------------

def dump_one_lmdb_to_jsonl(
    lmdb_path: str,
    model,
    processor,
    out_jsonl: str,
    batch_size: int = 1,
):
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    num_samples = _read_num_samples(env)
    dataset_name = os.path.basename(os.path.normpath(lmdb_path))

    skipped = 0
    prompt = "Extract the text in the image. Directly output all recognizable text without explanation or description."

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        with env.begin(write=False) as txn:
            for idx in tqdm(range(1, num_samples + 1), desc=f"HunyuanOCR | {dataset_name}"):
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

                try:
                    messages = [
                        {"role": "system", "content": ""},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_img},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ]
                    texts = [
                        processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    ]
                    inputs = processor(
                        text=texts, images=pil_img, padding=True, return_tensors="pt"
                    )
                    device = next(model.parameters()).device
                    inputs = inputs.to(device)

                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs, max_new_tokens=4096, do_sample=False
                        )
                    input_ids = inputs.input_ids if "input_ids" in inputs else inputs.inputs
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(input_ids, generated_ids)
                    ]
                    output_texts = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    recognized_text = (
                        output_texts[0].strip().replace("\n", " ").replace("\r", " ")
                    )
                except Exception:
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

    model_name = "tencent/HunyuanOCR"
    out_dir = "./hunyuanocr_results"
    os.makedirs(out_dir, exist_ok=True)

    model, processor = load_model(model_name)

    for p in lmdb_list:
        ds = os.path.basename(os.path.normpath(p))
        out_jsonl = os.path.join(out_dir, f"{ds}.jsonl")
        summary = dump_one_lmdb_to_jsonl(p, model, processor, out_jsonl)
        print("[DUMP]", summary)
