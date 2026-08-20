# infer_intern.py - InternVL3.5-8B (lmdeploy), LMDB -> JSONL
import os
import json
import lmdb
import cv2
import numpy as np
import unicodedata
from tqdm import tqdm
from PIL import Image
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image


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

def load_model(model_name="OpenGVLab/InternVL3_5-8B"):
    print(f"Loading {model_name} ...")
    pipe = pipeline(
        model_name,
        backend_config=PytorchEngineConfig(session_len=32768, tp=1),
    )
    return pipe


# -------------------- Single LMDB inference --------------------

def dump_one_lmdb_to_jsonl(
    lmdb_path: str,
    pipe,
    out_jsonl: str,
    batch_size: int = 1,
):
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    num_samples = _read_num_samples(env)
    dataset_name = os.path.basename(os.path.normpath(lmdb_path))

    prompt = (
        "Please directly output all original readable text in the image. "
        "Do not include any explanation or description. Only output the recognized text."
    )
    skipped = 0

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        with env.begin(write=False) as txn:
            for idx in tqdm(range(1, num_samples + 1), desc=f"InternVL3.5 | {dataset_name}"):
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
                    response = pipe((prompt, pil_img))
                    recognized_text = response.text.strip()
                    recognized_text = recognized_text.replace("\n", " ").replace("\r", " ").strip()
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

    model_name = "OpenGVLab/InternVL3_5-8B"
    out_dir = "./internvl_results"
    os.makedirs(out_dir, exist_ok=True)

    pipe = load_model(model_name)

    for p in lmdb_list:
        ds = os.path.basename(os.path.normpath(p))
        out_jsonl = os.path.join(out_dir, f"{ds}.jsonl")
        summary = dump_one_lmdb_to_jsonl(p, pipe, out_jsonl)
        print("[DUMP]", summary)
