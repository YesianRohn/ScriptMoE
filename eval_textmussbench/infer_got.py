# infer_got.py - GOT-OCR2.0, LMDB -> JSONL
import os
import json
import lmdb
import cv2
import numpy as np
import tempfile
import unicodedata
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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


def _decode_image_bin_to_bgr(image_bin: bytes):
    arr = np.frombuffer(image_bin, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# -------------------- Model loading --------------------

def load_model(model_name="ucaslcl/GOT-OCR2_0"):
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    ).eval().cuda()
    return model, tokenizer


# -------------------- Single LMDB inference --------------------

def dump_one_lmdb_to_jsonl(
    lmdb_path: str,
    model,
    tokenizer,
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
    # GOT-OCR chat interface requires an image file path
    tmp_dir = tempfile.mkdtemp(prefix="gotocr_")

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        with env.begin(write=False) as txn:
            for idx in tqdm(range(1, num_samples + 1), desc=f"GOT-OCR2 | {dataset_name}"):
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

                img = _decode_image_bin_to_bgr(image_bin)
                if img is None or img.size == 0:
                    skipped += 1
                    fw.write(json.dumps({
                        "dataset": dataset_name, "lmdb_path": lmdb_path,
                        "idx": idx, "gt": label,
                        "pred": "", "pred_score": None,
                        "skipped": True, "skip_reason": "decode_failed",
                    }, ensure_ascii=False) + "\n")
                    continue

                # GOT-OCR chat needs a file path
                tmp_img_path = os.path.join(tmp_dir, f"{idx:09d}.png")
                cv2.imwrite(tmp_img_path, img)

                res = model.chat(tokenizer, tmp_img_path, ocr_type="ocr")
                recognized_text = str(res).replace("\n", " ").replace("\r", " ").strip()

                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)

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
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"lmdb_path": lmdb_path, "dataset": dataset_name,
            "num_samples": num_samples, "skipped": skipped}


# -------------------- main --------------------

if __name__ == "__main__":
    lmdb_list = [

    ]

    model_name = "ucaslcl/GOT-OCR2_0"
    out_dir = "./gotocr2_results"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer = load_model(model_name)

    for p in lmdb_list:
        ds = os.path.basename(os.path.normpath(p))
        out_jsonl = os.path.join(out_dir, f"{ds}.jsonl")
        summary = dump_one_lmdb_to_jsonl(p, model, tokenizer, out_jsonl)
        print("[DUMP]", summary)
