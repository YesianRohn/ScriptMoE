# infer_ppocrv5_mlt.py - PP-OCRv5 MLT, LMDB -> JSONL
import os
import json
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
import unicodedata
from paddleocr import TextRecognition


def _read_num_samples(env: lmdb.Environment) -> int:
    with env.begin(write=False) as txn:
        v = txn.get(b"num-samples")
        if v is None:
            raise RuntimeError("LMDB missing key: num-samples")
        return int(v.decode("utf-8"))


def _get_item(txn: lmdb.Transaction, idx: int):
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


def dump_one_lmdb_to_jsonl(
    lmdb_path: str,
    model: TextRecognition,
    out_jsonl: str,
    batch_size: int = 16,
):
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    num_samples = _read_num_samples(env)

    dataset_name = os.path.basename(os.path.normpath(lmdb_path))

    skipped = 0
    imgs, metas = [], []

    def flush_batch(fw):
        nonlocal imgs, metas
        if not imgs:
            return
        outputs = model.predict(input=imgs, batch_size=len(imgs))
        for out, meta in zip(outputs, metas):
            rec_text = out.get("rec_text", "")
            rec_text = rec_text.replace("\n", " ").replace("\r", " ").strip()
            rec_score = out.get("rec_score", None)
            row = {
                "dataset": dataset_name,       # used for metric aggregation
                "lmdb_path": lmdb_path,        # traceable source
                "idx": meta["idx"],            # 1-based index in lmdb
                "gt": meta["gt"],
                "pred": rec_text,
                "pred_score": rec_score,
                "skipped": False,
            }
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")

        imgs, metas = [], []

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        with env.begin(write=False) as txn:
            for idx in tqdm(range(1, num_samples + 1), desc=f"Dump {dataset_name}"):
                image_bin, label = _get_item(txn, idx)
                if image_bin is None:
                    skipped += 1
                    fw.write(json.dumps({
                        "dataset": dataset_name,
                        "lmdb_path": lmdb_path,
                        "idx": idx,
                        "gt": label if label is not None else "",
                        "pred": "",
                        "pred_score": None,
                        "skipped": True,
                        "skip_reason": "missing_image_or_label",
                    }, ensure_ascii=False) + "\n")
                    continue

                img = _decode_image_bin_to_bgr(image_bin)
                if img is None or img.size == 0:
                    skipped += 1
                    fw.write(json.dumps({
                        "dataset": dataset_name,
                        "lmdb_path": lmdb_path,
                        "idx": idx,
                        "gt": label,
                        "pred": "",
                        "pred_score": None,
                        "skipped": True,
                        "skip_reason": "decode_failed",
                    }, ensure_ascii=False) + "\n")
                    continue

                imgs.append(img)
                metas.append({"idx": idx, "gt": label})

                if len(imgs) >= batch_size:
                    flush_batch(fw)

            flush_batch(fw)

    env.close()
    return {"lmdb_path": lmdb_path, "dataset": dataset_name, "num_samples": num_samples, "skipped": skipped}



if __name__ == "__main__":
    lmdb_list = [
    ]

    model_name = "cyrillic_PP-OCRv5_mobile_rec"  # cyrillic for example
    out_dir = "./ppocrv5_results"
    os.makedirs(out_dir, exist_ok=True)

    # One JSONL per LMDB for clarity
    for p in lmdb_list:
        ds = os.path.basename(os.path.normpath(p))
        out_jsonl = os.path.join(out_dir, f"{ds}.jsonl")
        summary = dump_one_lmdb_to_jsonl(
            p, model=TextRecognition(model_name=model_name), out_jsonl=out_jsonl, batch_size=16
        )
        print("[DUMP]", summary)
