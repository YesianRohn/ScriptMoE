# All-in-One Multilingual Scene Text Recognition with Script-aware Mixture-of-Experts

## Directory Structure

```
ScriptMoE/
├── OpenOCR/              # Training code (OpenOCR framework; ScriptMoE model/loss/postprocess)
├── E2EOCR/               # End-to-end deployment & inference (detector + ScriptMoE recognizer, self-contained)
│   ├── infer.py          #   End-to-end inference CLI (detection + recognition)
│   ├── modeling.py       #   Self-contained PyTorch model (SVTRv2 + ScriptMoE decoder)
│   ├── postprocess.py    #   AR label decoding
│   ├── visualize.py      #   Text-box cropping / visualization
│   ├── model.safetensors #   Recognizer weights
│   └── assets/dict.txt   #   Multilingual character dictionary
├── run_cc_ocr_mlt.py     # End-to-end evaluation script on the CC-OCR-MLT dataset
├── CC-OCR-MLT/           # Test dataset & official evaluator
│   ├── tsv/              #   10 languages, <Lang>_150.tsv (base64 images + GT)
│   └── eval/             #   Official evaluator (main.py + evaluator/)
└── README.md
```

Roles of the four modules:

| Module | Purpose |
|--------|---------|
| `OpenOCR/` | **Training**: train the ScriptMoE recognizer with the OpenOCR framework |
| `E2EOCR/` | **Deployment/Inference**: detector + ScriptMoE recognizer pipeline for arbitrary images |
| `run_cc_ocr_mlt.py` | **Evaluation**: run the end-to-end pipeline on CC-OCR-MLT and compute metrics |
| `CC-OCR-MLT/` | **Data/Evaluator**: test-set TSVs + official evaluation code |

---


## Installation

```bash
# Recognizer + end-to-end inference dependencies
cd E2EOCR
pip install -r requirements.txt

# Training dependencies (OpenOCR framework)
cd ../OpenOCR
pip install -r requirements.txt
```

Core dependencies: `torch>=2.0`, `torchvision`, `safetensors`, `opencv-python`, `Pillow`.

Detectors are optional; install as needed:
- `openocr` (default): `pip install openocr-python`
- `ppv5` / `ppv6`: `pip install "paddleocr>=3.7.0" paddlepaddle` (PP-OCRv6 requires `paddleocr>=3.7.0`)

---

## 1. End-to-End Inference (E2EOCR)

Run "detection + ScriptMoE recognition" on a single image or a whole directory. The recognizer weights (`model.safetensors`) and dictionary (`assets/dict.txt`) are shipped in the folder.

```bash
cd E2EOCR

# Default: openocr detector
python infer.py --image path/to/img.jpg

# PP-OCRv5 detector
python infer.py --image path/to/img.jpg --det ppv5

# Recognize the whole image only (no detection; for pre-cropped text lines)
python infer.py --image path/to/crop.jpg --det rec_only

# Batch a directory and write JSON
python infer.py --image path/to/dir --output out.json
```

Common arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--det` | `openocr` | Detector: `openocr` / `ppv5` / `ppv6` / `rec_only` |
| `--drop_score` | `0.5` | Drop boxes whose recognition score is below this value |
| `--rec_batch_num` | `8` | Recognition batch size |
| `--use_gpu` | `auto` | `auto` / `true` / `false` |
| `--max_ratio` | `20` | Max width/height ratio for the recognizer |

> Note: the recognizer weights, dictionary, and detector mode (server) are hard-coded in the script; no CLI flags are needed for them.

---

## 2. CC-OCR-MLT End-to-End Evaluation (run_cc_ocr_mlt.py)

Run the end-to-end pipeline on the CC-OCR multilingual benchmark (10 languages, 150 images each) and invoke the official evaluator to report F1.

Pipeline: read local `CC-OCR-MLT/tsv/<Lang>_150.tsv` → decode base64 images → E2EOCR inference → filter boxes by `det_thresh` / `rec_thresh` → concatenate the response → write GT and prediction JSONs → call `CC-OCR-MLT/eval/main.py` to produce `summary.md`.

```bash
cd ScriptMoE

# Reproduce the paper's end-to-end setting: PP-OCRv5 detector + det_thresh=0.0 / rec_thresh=0.7
python run_cc_ocr_mlt.py --det ppv5 --det_thresh 0.0 --rec_thresh 0.7

# A subset of languages / limited count (for debugging)
python run_cc_ocr_mlt.py --det ppv5 --languages Korean Japanese --max_per_lang 20

# Inference only, no evaluation
python run_cc_ocr_mlt.py --det ppv5 --no_eval
```

Common arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--det` | `openocr` | Detector; use `ppv5` to reproduce the paper |
| `--det_thresh` | `0.0` | Detection score threshold |
| `--rec_thresh` | `0.7` | Recognition score threshold |
| `--languages` | all 10 | Subset of languages |
| `--max_per_lang` | `150` | Max images per language |
| `--work_dir` | `./cc_ocr_eval` | Output root directory |
| `--exp_name` | auto `det_d<thr>_r<thr>` | Experiment name |
| `--no_eval` | off | Inference only, no evaluation |
| `--show_pred` | off | Also print predicted text in progress logs |

Results are written to `cc_ocr_eval/summary.md` and a `status.json` under each experiment directory.

---

## 3. Training (OpenOCR)

Training is built on the `OpenOCR/` framework. The ScriptMoE config lives at `OpenOCR/configs/rec/scriptmoe/svtrv2_scriptmoe_mlt.yml`, and the relevant implementation files are:

- Model: `OpenOCR/openrec/modeling/decoders/scriptmoe_decoder.py`
- Loss: `OpenOCR/openrec/losses/scriptmoe_loss.py`
- Postprocess: `OpenOCR/openrec/postprocess/scriptmoe_postprocess.py`
- Encoder: `OpenOCR/openrec/modeling/encoders/svtrv2_lnconv_two33.py`

```bash
cd OpenOCR

# Single-node multi-GPU training (example; adjust GPU count and port as needed)
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train_rec.py --c configs/rec/scriptmoe/svtrv2_scriptmoe_mlt.yml
```

Before training, fill in the training/validation dataset paths in the yml (`Train.dataset.data_dir_list` / `Eval.dataset.data_dir_list`).

**Key configuration (aligned with the paper)**:

| Item | STR setting | End-to-end setting |
|------|-------------|--------------------|
| `max_text_length` | 25 | 100 |
| `max_ratio` | 8 | 20 |

**Training recipe** (paper appendix): AdamW (weight decay 0.05), peak learning rate `6.5e-4`, global batch size 1024 (8×128), OneCycleLR with a 1.5-epoch linear warmup, 2 epochs total; label smoothing 0.1. The end-to-end variant is re-trained with `max_text_length=100` to handle line-level inputs.

---

## Datasets

- **Training**: real English/Chinese data (Union14M, BCTR) + a small amount of real multilingual data (MLT2019) + large-scale synthetic data **TextMuSS-10M** (10 scripts, 229 languages, 1M per script).
- **STR evaluation**: **TextMuSS-Bench** (extends MLT2019 with newly collected Russian / Thai / Tibetan; 10,899 real images total).
- **End-to-end evaluation**: CC-OCR multilingual task; this repo ships `CC-OCR-MLT/tsv/` with 150 images per language for 10 languages.

---

## Results at a Glance

**TextMuSS-Bench (per-script word accuracy %, excerpt)**

| Method | Arabic | Chinese | Japanese | Korean | Thai | Tibetan | Avg |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SVTRv2-AR | 75.11 | 94.15 | 69.36 | 85.86 | 69.60 | 86.80 | 80.75 |
| **ScriptMoE** | **78.09** | **95.38** | **71.21** | **87.19** | **72.00** | **88.76** | **82.06** |

**CC-OCR end-to-end (F1 %, excerpt)**

| Method | Korean | Japanese | Russian | Total |
|--------|:---:|:---:|:---:|:---:|
| PP-OCRv5 MLT | 78.58 | 76.13 | 49.67 | 65.71 |
| Qwen2.5-VL-72B | 85.36 | 76.27 | 71.09 | 79.68 |
| **PP-OCRv5 Det + ScriptMoE** | **92.33** | **89.43** | **79.22** | **80.89** |

---

## Citation

```bibtex
@inproceedings{scriptmoe,
  title  = {All-in-One Multilingual Scene Text Recognition with Script-aware Mixture-of-Experts},
  author = {Anonymous},
  year   = {2026}
}
```
