<p align="center">
  <h3 align="center">All-in-One Multilingual Scene Text Recognition with Script-aware Mixture-of-Experts</h3>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.24058">
    <img src="https://img.shields.io/badge/arXiv-2609.24058-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/papers/2609.24058">
    <img src="https://img.shields.io/badge/Hugging%20Face-Paper-ffcc00.svg" alt="Hugging Face Paper">
  </a>
  <a href="https://huggingface.co/spaces/Yesianrohn/MultilingualOCR-Demo">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Demo-yellow.svg" alt="Demo">
  </a>
</p>

**Xingsong Ye, Yongkun Du, Jiaxin Zhang, Zhixian Li, Chong Sun, Chen Li, Jing Lyu, Lianwen Jin, Zhineng Chen**

---

## Highlights

- 🌍 **All-in-one multilingual STR** across 10 writing scripts and 229 languages.
- 🧩 **Script-aware Mixture-of-Experts** with top-2 script-aligned routing.
- 🔗 **Shared expert** for cross-script knowledge transfer.
- 🏗️ **Single visual encoder** shared across all scripts.
- 🧪 **TextMuSS-10M**: 10 scripts, 229 languages, 1M synthetic samples per script.
- 📊 **TextMuSS-Bench**: 10 scripts and 10,899 real images.
- 🚀 **End-to-end OCR** integration with PP-OCRv5.
- 🔬 Complete training, inference, and evaluation code is provided in this repository.

---

## Resources

| Resource | Description |
|---|---|
| 📄 [Paper](https://arxiv.org/abs/2609.24058) | Research paper on arXiv |
| 🤗 [Hugging Face Paper](https://huggingface.co/papers/2609.24058) | Hugging Face paper page |
| 💻 [Code](https://github.com/YesianRohn/ScriptMoE) | Official implementation |
| 🚀 [Demo](https://huggingface.co/spaces/Yesianrohn/MultilingualOCR-Demo) | Interactive multilingual OCR demo |
| 🧪 [TextMuSS-10M](https://huggingface.co/datasets/Yesianrohn/TextMuSS-10M) | Large-scale synthetic training dataset |
| 📊 [TextMuSS-Bench](https://huggingface.co/datasets/Yesianrohn/TextMuSS-Bench) | Multilingual STR benchmark |

---

## Method

ScriptMoE consists of a shared visual encoder followed by a sparse script-aware MoE decoder.

Given an input text image, an image-level router predicts the relevant scripts and dispatches the input to the **top-2 script-aligned experts**. A shared expert is additionally used to absorb knowledge shared across different writing systems.

Conceptually:

```text
                 Input Image
                     │
                     ▼
          ┌─────────────────────┐
          │ Shared Visual       │
          │ Encoder (SVTRv2)    │
          └──────────┬──────────┘
                     │
                     ▼
             Image-level Router
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       Expert 1   Expert 2   Shared Expert
       Script-A   Script-B   Cross-script
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
             Sparse MoE Decoder
                     │
                     ▼
              Recognized Text
```

---

## Directory Structure

```text
ScriptMoE/
├── OpenOCR/
│   ├── configs/
│   │   └── rec/
│   │       └── scriptmoe/
│   │           └── svtrv2_scriptmoe_mlt.yml
│   └── openrec/
│       ├── modeling/
│       ├── losses/
│       └── postprocess/
│
├── E2EOCR/
│   ├── infer.py
│   ├── modeling.py
│   ├── postprocess.py
│   ├── visualize.py
│   ├── model.safetensors
│   └── assets/
│       └── dict.txt
│
├── eval_textmussbench/
├── Eval-TextMuSS-Bench/
├── CC-OCR-MLT/
├── run_cc_ocr_mlt.py
└── README.md
```

### Main components

| Module | Purpose |
|---|---|
| `OpenOCR/` | Training code and ScriptMoE implementation |
| `E2EOCR/` | Self-contained end-to-end inference pipeline |
| `eval_textmussbench/` | TextMuSS-Bench evaluation |
| `Eval-TextMuSS-Bench/` | Evaluation utilities |
| `run_cc_ocr_mlt.py` | End-to-end CC-OCR-MLT evaluation |
| `CC-OCR-MLT/` | CC-OCR-MLT test data and official evaluator |

---

## Installation

### End-to-end inference

```bash
cd E2EOCR
pip install -r requirements.txt
```

Core dependencies include:

```text
torch >= 2.0
torchvision
safetensors
opencv-python
Pillow
```

Optional detectors:

```bash
# OpenOCR detector
pip install openocr-python

# PP-OCRv5 / PP-OCRv6
pip install "paddleocr>=3.7.0" paddlepaddle
```

### Training

```bash
cd OpenOCR
pip install -r requirements.txt
```

---

## End-to-End Inference

The `E2EOCR/` directory provides a self-contained inference pipeline consisting of text detection followed by ScriptMoE recognition.

The recognizer weights and multilingual character dictionary are already included.

### Single image

```bash
cd E2EOCR

python infer.py --image path/to/image.jpg
```

### PP-OCRv5 detector

```bash
python infer.py \
    --image path/to/image.jpg \
    --det ppv5
```

### Recognition only

For an already cropped text line:

```bash
python infer.py \
    --image path/to/crop.jpg \
    --det rec_only
```

### Batch inference

```bash
python infer.py \
    --image path/to/image_directory \
    --output out.json
```

### Main arguments

| Argument | Default | Description |
|---|---:|---|
| `--det` | `openocr` | Detector: `openocr`, `ppv5`, `ppv6`, or `rec_only` |
| `--drop_score` | `0.5` | Remove recognition results below this score |
| `--rec_batch_num` | `8` | Recognition batch size |
| `--use_gpu` | `auto` | `auto`, `true`, or `false` |
| `--max_ratio` | `20` | Maximum width/height ratio |

---

## Training

Training is implemented on top of the OpenOCR framework.

The main ScriptMoE configuration is:

```text
OpenOCR/configs/rec/scriptmoe/svtrv2_scriptmoe_mlt.yml
```

Important implementation files include:

```text
OpenOCR/openrec/modeling/decoders/scriptmoe_decoder.py
OpenOCR/openrec/losses/scriptmoe_loss.py
OpenOCR/openrec/postprocess/scriptmoe_postprocess.py
OpenOCR/openrec/modeling/encoders/svtrv2_lnconv_two33.py
```

Before training, update the training and validation dataset paths in:

```yaml
Train.dataset.data_dir_list
Eval.dataset.data_dir_list
```

### Example

```bash
cd OpenOCR

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    tools/train_rec.py \
    --c configs/rec/scriptmoe/svtrv2_scriptmoe_mlt.yml
```

### Training recipe

The main training recipe follows the paper:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 0.05 |
| Peak learning rate | `6.5e-4` |
| Global batch size | 1024 |
| GPUs | 8 × 128 |
| Scheduler | OneCycleLR |
| Warmup | 1.5 epochs |
| Total training | 2 epochs |
| Label smoothing | 0.1 |

For standard STR evaluation:

```text
max_text_length = 25
max_ratio = 8
```

For end-to-end recognition:

```text
max_text_length = 100
max_ratio = 20
```

---


## Evaluation

### TextMuSS-Bench

The repository provides evaluation code for TextMuSS-Bench.

```bash
python eval_textmussbench/xxx.py
```

Please refer to the scripts under:

```text
eval_textmussbench/
Eval-TextMuSS-Bench/
```

for the corresponding evaluation settings.

### CC-OCR-MLT

To reproduce the end-to-end evaluation reported in the paper:

```bash
python run_cc_ocr_mlt.py \
    --det ppv5 \
    --det_thresh 0.0 \
    --rec_thresh 0.7
```

For a smaller subset:

```bash
python run_cc_ocr_mlt.py \
    --det ppv5 \
    --languages Korean Japanese \
    --max_per_lang 20
```

Inference without evaluation:

```bash
python run_cc_ocr_mlt.py \
    --det ppv5 \
    --no_eval
```

---

## Results

### TextMuSS-Bench

ScriptMoE achieves **82.06% average word accuracy** on TextMuSS-Bench.

| Method | Arabic | Chinese | Japanese | Korean | Thai | Tibetan | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVTRv2-AR | 75.11 | 94.15 | 69.36 | 85.86 | 69.60 | 86.80 | 80.75 |
| **ScriptMoE** | **78.09** | **95.38** | **71.21** | **87.19** | **72.00** | **88.76** | **82.06** |

### CC-OCR End-to-End

Replacing the recognizer in PP-OCRv5 with ScriptMoE gives:

| Method | Korean | Japanese | Russian | Total F1 |
|---|---:|---:|---:|---:|
| PP-OCRv5 MLT | 78.58 | 76.13 | 49.67 | 65.71 |
| Qwen2.5-VL-72B | 85.36 | 76.27 | 71.09 | 79.68 |
| **PP-OCRv5 Det + ScriptMoE** | **92.33** | **89.43** | **79.22** | **80.89** |

---

## Citation

If you find ScriptMoE, TextMuSS-10M, or TextMuSS-Bench useful in your research, please cite:

```bibtex
@article{ye2026scriptmoe,
  title   = {All-in-One Multilingual Scene Text Recognition with Script-aware Mixture-of-Experts},
  author  = {Ye, Xingsong and Du, Yongkun and Zhang, Jiaxin and Li, Zhixian and Sun, Chong and Li, Chen and Lyu, Jing and Jin, Lianwen and Chen, Zhineng},
  journal = {arXiv preprint arXiv:2609.24058},
  year    = {2026}
}
```

---

## License

The code in this repository is released under the license specified in the repository.

Please check the licenses of the individual datasets and third-party components before redistribution or commercial use.
