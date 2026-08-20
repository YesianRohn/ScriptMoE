# infer_ppocrv6.py - PP-OCRv6_medium_rec (transformers engine), LMDB -> JSONL
import os
import json
import lmdb
import numpy as np
import unicodedata
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from paddleocr import TextRecognition


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


def _decode_image_bin_to_ndarray(image_bin: bytes):
    """Decode image bytes to an RGB numpy.ndarray for TextRecognition.predict."""
    try:
        img = Image.open(BytesIO(image_bin)).convert("RGB")
        return np.array(img)
    except Exception:
        return None


# -------------------- Model loading --------------------

def load_model(model_name="PP-OCRv6_medium_rec"):
    print(f"Loading {model_name} (engine=transformers) ...")
    model = TextRecognition(
        model_name=model_name,
        engine="transformers",
    )
    return model


# -------------------- Single LMDB inference --------------------

def dump_one_lmdb_to_jsonl(
    lmdb_path: str,
    model,
    out_jsonl: str,
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
            for idx in tqdm(range(1, num_samples + 1), desc=f"PP-OCRv6 | {dataset_name}"):
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

                img = _decode_image_bin_to_ndarray(image_bin)
                if img is None:
                    skipped += 1
                    fw.write(json.dumps({
                        "dataset": dataset_name, "lmdb_path": lmdb_path,
                        "idx": idx, "gt": label,
                        "pred": "", "pred_score": None,
                        "skipped": True, "skip_reason": "decode_failed",
                    }, ensure_ascii=False) + "\n")
                    continue

                recognized_text = ""
                pred_score = None
                try:
                    output = model.predict(input=img, batch_size=1)
                    for res in output:
                        recognized_text = res.get("rec_text", "") or ""
                        pred_score = res.get("rec_score", None)
                        break
                    recognized_text = recognized_text.replace("\n", " ").replace("\r", " ").strip()
                except Exception as e:
                    recognized_text = ""
                    pred_score = None

                row = {
                    "dataset": dataset_name,
                    "lmdb_path": lmdb_path,
                    "idx": idx,
                    "gt": label,
                    "pred": recognized_text,
                    "pred_score": pred_score,
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

    model_name = "PP-OCRv6_medium_rec"
    out_dir = "./ppocrv6_results"
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(model_name)

    for p in lmdb_list:
        ds = os.path.basename(os.path.normpath(p))
        out_jsonl = os.path.join(out_dir, f"{ds}.jsonl")
        summary = dump_one_lmdb_to_jsonl(p, model, out_jsonl)
        print("[DUMP]", summary)
