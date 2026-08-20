# infer_qwen3_5.py - Qwen3.5-VL (vLLM), LMDB -> JSONL
import os
import json
import lmdb
import cv2
import numpy as np
import unicodedata
import tempfile
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


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

def load_model(model_path, tp_size=1):
    print(f"Loading {model_path} ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 1},
    )
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
    )
    return llm, processor


# -------------------- Single LMDB inference --------------------

def dump_one_lmdb_to_jsonl(
    lmdb_path: str,
    llm,
    processor,
    out_jsonl: str,
    batch_size: int = 64,
):
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    num_samples = _read_num_samples(env)
    dataset_name = os.path.basename(os.path.normpath(lmdb_path))

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512, stop=["<|im_end|>"])
    prompt_text = (
        "Please directly output all original readable text in the image. "
        "Do not include any explanation or description. Only output the recognized text."
    )
    skipped = 0

    # vLLM needs image file paths, so create a temporary directory
    tmp_dir = tempfile.mkdtemp(prefix="qwen_vl_")

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        with env.begin(write=False) as txn:
            batch_inputs = []
            batch_metas = []
            batch_tmp_files = []

            for idx in tqdm(range(1, num_samples + 1), desc=f"Qwen3.5-VL | {dataset_name}"):
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

                # vLLM needs an image path
                tmp_img_path = os.path.join(tmp_dir, f"{idx:09d}.png")
                pil_img.save(tmp_img_path)
                batch_tmp_files.append(tmp_img_path)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": tmp_img_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                image_inputs, _ = process_vision_info(messages)

                batch_inputs.append({
                    "prompt": text,
                    "multi_modal_data": {"image": image_inputs},
                })
                batch_metas.append({"idx": idx, "gt": label})

                if len(batch_inputs) >= batch_size:
                    # flush batch
                    try:
                        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
                        for out, meta in zip(outputs, batch_metas):
                            rec_text = out.outputs[0].text.strip()
                            rec_text = rec_text.replace("\n", " ").replace("\r", " ").strip()
                            row = {
                                "dataset": dataset_name,
                                "lmdb_path": lmdb_path,
                                "idx": meta["idx"],
                                "gt": meta["gt"],
                                "pred": rec_text,
                                "pred_score": None,
                                "skipped": False,
                            }
                            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                    except Exception:
                        for meta in batch_metas:
                            fw.write(json.dumps({
                                "dataset": dataset_name, "lmdb_path": lmdb_path,
                                "idx": meta["idx"], "gt": meta["gt"],
                                "pred": "", "pred_score": None,
                                "skipped": True, "skip_reason": "inference_error",
                            }, ensure_ascii=False) + "\n")

                    # Clean up temporary files
                    for f in batch_tmp_files:
                        if os.path.exists(f):
                            os.remove(f)
                    batch_inputs, batch_metas, batch_tmp_files = [], [], []

            # flush remaining
            if batch_inputs:
                try:
                    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
                    for out, meta in zip(outputs, batch_metas):
                        rec_text = out.outputs[0].text.strip()
                        rec_text = rec_text.replace("\n", " ").replace("\r", " ").strip()
                        row = {
                            "dataset": dataset_name,
                            "lmdb_path": lmdb_path,
                            "idx": meta["idx"],
                            "gt": meta["gt"],
                            "pred": rec_text,
                            "pred_score": None,
                            "skipped": False,
                        }
                        fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                except Exception:
                    for meta in batch_metas:
                        fw.write(json.dumps({
                            "dataset": dataset_name, "lmdb_path": lmdb_path,
                            "idx": meta["idx"], "gt": meta["gt"],
                            "pred": "", "pred_score": None,
                            "skipped": True, "skip_reason": "inference_error",
                        }, ensure_ascii=False) + "\n")

                for f in batch_tmp_files:
                    if os.path.exists(f):
                        os.remove(f)

    env.close()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"lmdb_path": lmdb_path, "dataset": dataset_name,
            "num_samples": num_samples, "skipped": skipped}


# -------------------- main --------------------

if __name__ == "__main__":
    lmdb_list = [

    ]

    model_path = "Qwen/Qwen3.5-9B"  # Fill in the Qwen3.5-VL model path
    tp_size = 1      # Adjust according to GPU count
    out_dir = "./qwen3_5_9b_results"
    os.makedirs(out_dir, exist_ok=True)

    llm, processor = load_model(model_path, tp_size)

    for p in lmdb_list:
        ds = os.path.basename(os.path.normpath(p))
        out_jsonl = os.path.join(out_dir, f"{ds}.jsonl")
        summary = dump_one_lmdb_to_jsonl(p, llm, processor, out_jsonl, batch_size=64)
        print("[DUMP]", summary)
