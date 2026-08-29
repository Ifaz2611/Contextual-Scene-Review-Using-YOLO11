# Getting Started — Training YOLO11 on the Included Dataset

> **Project:** Contextual Scene Review (YOLO11) — `person` / `person_with_object` detector
> **Stack:** `ultralytics>=8.3` + PyTorch, YOLO11 variants (n/s/m/l)

This guide shows how to set up the environment and train YOLO11 using the dataset already included in `dataset/` (YOLO format, train/valid/test splits).

---

## 1. Prerequisites

| Requirement | Version / Notes |
|---|---|
| OS | Windows 10/11 (PowerShell), Linux or macOS also works |
| Python | 3.10 – 3.12 recommended (tested with 3.10+) |
| GPU (optional) | NVIDIA GPU + CUDA for fast training. CPU works but is ~10-20x slower |
| Disk | ~2 GB for dataset + ~5 GB for `runs/` outputs + weights |
| RAM | 8 GB+ (16 GB recommended for `imgsz=832`, `batch=8`) |

Verify Python:

```powershell
python --version
pip --version
nvidia-smi  # optional — check GPU + CUDA
```

---

## 2. Dataset Overview

Dataset is already bundled at `dataset/` (also ignored by `.gitignore` for releases):

```
dataset/
├── data.yaml              # dataset config (now aligned to English names)
├── README.dataset.txt     # Roboflow export info (CC BY 4.0)
├── README.roboflow.txt
├── train/
│   ├── images/  (2512 images)
│   └── labels/  (2512 .txt, YOLO format: class cx cy w h normalized)
├── valid/
│   ├── images/  (314 images)
│   └── labels/  (314 .txt)
└── test/
    ├── images/  (314 images)
    └── labels/  (314 .txt)
```

**Classes (contract — do not rename silently):**

| ID | Name | Meaning |
|---|---|---|
| 0 | `person` | Person without object context |
| 1 | `person_with_object` | Person with object — for human review workflow |

> **Note:** Both `dataset/data.yaml:6` and `config/data.yaml:6` now use the same English names (`person`, `person_with_object`). You can train with either `config/data.yaml` (recommended, after fixing `path` in §3.2) or `dataset/data.yaml`.

Label example `dataset/train/labels/train_0001.txt`:
```
0 0.1865234375 0.5947265625 0.103515625 0.33203125
0 0.091796875 0.6162109375 0.095703125 0.3310546875
1 0.512 0.401 0.120 0.250
```

Pre-processing already applied on export (Roboflow): auto-orient + resize to `512x512` (Stretch). No augmentation baked in — augmentation is done at train time via `config/experiments.yaml`.

---

## 3. Environment Setup

### 3.1 Create virtual environment & install dependencies

**Windows PowerShell:**

```powershell
# 1. Clone / open project (if not already)
# cd "C:\Users\zahin\Downloads\behavior tracking-YOLO11\behavior tracking using YOLO11"

# 2. Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# If execution policy blocks it:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Upgrade pip and install
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt:3` installs:
```
ultralytics>=8.3,<9.0
PyYAML>=6.0,<7.0
Pillow>=10.0,<13.0
```
PyTorch will be pulled as a dependency of `ultralytics`. For a specific CUDA build, install PyTorch first from https://pytorch.org/get-started/locally/ then `pip install -r requirements.txt`.

Verify install:

```powershell
python -c "from ultralytics import YOLO; print(YOLO)"
python -c "import torch; print(torch.cuda.is_available())"
```

### 3.2 Fix dataset path in `config/data.yaml`

`config\data.yaml:1` currently contains:

```yaml
path: ../data/dataset   # ← WRONG for this repo layout
train: train/images
val: valid/images
test: test/images
names:
  0: person
  1: person_with_object
```

Because the included dataset lives at `./dataset`, not `../data/dataset`, training will fail with `dataset not found` unless you fix it.

**Fix — pick ONE:**

**Option A — Recommended (relative to project root):**

Edit `config/data.yaml` to:

```yaml
path: dataset
train: train/images
val: valid/images
test: test/images
names:
  0: person
  1: person_with_object
```

**Option B — Absolute path (most robust on Windows):**

```yaml
path: C:/Users/zahin/Downloads/behavior tracking-YOLO11/behavior tracking using YOLO11/dataset
train: train/images
val: valid/images
test: test/images
names:
  0: person
  1: person_with_object
```

> Use forward slashes `/` even on Windows — Ultralytics/YAML handles them correctly.

Validate the fix:

```powershell
Test-Path -LiteralPath "dataset\train\images"
Test-Path -LiteralPath "config\data.yaml"
Get-Content -LiteralPath "config\data.yaml"
```

Alternatively, you can bypass the file and pass an absolute dataset path via CLI (not recommended for reproducibility):

```powershell
python scripts/train.py --variant yolo11s --data "C:/absolute/path/to/dataset/data.yaml"
# but then you must also correct names in dataset/data.yaml first
```

---

## 4. Training

All training is driven by `scripts\train.py:11` + `config\experiments.yaml:1`.

### 4.1 Training defaults (`scripts\train.py:45`)

| Param | Value | Notes |
|---|---|---|
| `epochs` | 150 | Early stopping via `patience` |
| `batch` | 8 | Reduce to 4 if OOM |
| `imgsz` | 832 | Training image size |
| `optimizer` | auto | Ultralytics selects AdamW/SGD |
| `device` | auto | Set `--device 0` for GPU, `--device cpu` for CPU |
| `project` | runs/train | Output parent |
| `name` | `<variant>` | Subfolder |

### 4.2 Variants (`config\experiments.yaml:8`)

| Variant | Base weights | Patience | Completed epochs (reference) | Use case |
|---|---|---|---|---|
| `yolo11n` | `yolo11n.pt` | 35 | 109 | Fastest, lowest accuracy — prototyping |
| `yolo11s` | `yolo11s.pt` | 35 | 132 | **Recommended default** — good speed/accuracy |
| `yolo11s_preprocessing` | `yolo11s.pt` | 35 | 150 | Same as `s` + explicit aug: `fliplr=0.5, degrees=5, translate=0.1, scale=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, mosaic=0.7, mixup=0.05` |
| `yolo11m` | `yolo11m.pt` | 40 | 150 | Higher accuracy, needs more VRAM |
| `yolo11l` | `yolo11l.pt` | 40 | 150 | Highest accuracy, slowest |

### 4.3 Run training

**Default — YOLO11s (recommended first run):**

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train.py --variant yolo11s --data config/data.yaml
```

**Other variants:**

```powershell
python scripts/train.py --variant yolo11n --data config/data.yaml
python scripts/train.py --variant yolo11s_preprocessing --data config/data.yaml
python scripts/train.py --variant yolo11m --data config/data.yaml
python scripts/train.py --variant yolo11l --data config/data.yaml
```

**Explicit device selection:**

```powershell
# GPU 0
python scripts/train.py --variant yolo11s --data config/data.yaml --device 0

# CPU only (slow)
python scripts/train.py --variant yolo11s --data config/data.yaml --device cpu

# Multi-GPU (if available)
python scripts/train.py --variant yolo11s --data config/data.yaml --device 0,1
```

**Custom run name / output folder:**

```powershell
python scripts/train.py --variant yolo11s --data config/data.yaml --project runs/train --name my_first_run
# output → runs/train/my_first_run/
```

### 4.4 What happens during training

1. `YOLO("yolo11s.pt")` auto-downloads COCO-pretrained weights on first run (~20 MB for `s`).
2. Trains for up to `150` epochs with early stopping (`patience` 35/40).
3. Validates each epoch on `valid/images`.
4. Saves to `runs/train/<variant>/`:
   ```
   runs/train/yolo11s/
   ├── weights/
   │   ├── best.pt      ← USE THIS for inference
   │   └── last.pt
   ├── args.yaml
   ├── results.csv
   ├── confusion_matrix.png
   ├── F1_curve.png / PR_curve.png
   └── train_batch*.jpg  (augmented samples)
   ```

---

## 5. Using Your Own Dataset

To train on a different YOLO-format dataset:

1. Create the folder structure:
   ```
   my_data/
   ├── train/images/  + train/labels/
   ├── valid/images/  + valid/labels/
   └── test/images/   + test/labels/   (optional)
   ```
   Labels are `.txt` per image, one line per object: `<class_id> <cx> <cy> <w> <h>` normalized to 0-1.

2. Create `my_data.yaml`:
   ```yaml
   path: C:/absolute/path/to/my_data
   train: train/images
   val: valid/images
   test: test/images
   nc: 2
   names:
     0: person
     1: person_with_object
   ```

3. Train:
   ```powershell
   python scripts/train.py --variant yolo11s --data my_data.yaml
   ```

---

## 6. Inference & Validation After Training

**Run inference on an image/video/folder:**

```powershell
python scripts/inference.py --model runs/train/yolo11s/weights/best.pt --source path/to/image-or-video --confidence 0.35
# also works with:
# --source dataset/test/images
# --source path/to/video.mp4
# --source 0  (webcam)
```

Options (`scripts\inference.py:18`):

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | Path to `best.pt` |
| `--source` | (required) | Image, video, directory, or stream |
| `--confidence` | 0.35 | Confidence threshold 0.0–1.0 (lower = more detections) |
| `--project` | runs/predict | Output parent |
| `--name` | demo | Run subfolder |
| `--device` | auto | `cpu`, `0`, `0,1` etc. |

Output saved to `runs/predict/demo/` (annotated images/video).

**Validate on test split (YOLO CLI):**

```powershell
yolo val model=runs/train/yolo11s/weights/best.pt data=config/data.yaml imgsz=832
```

---

## 7. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Dataset not found` / `No such file` | `config/data.yaml` `path` still `../data/dataset` | Apply §3.2 fix — set `path: dataset` or absolute path |
| `FileNotFoundError: Model file not found` (inference) | Wrong `--model` path | Use `runs/train/<variant>/weights/best.pt` — check file exists with `Test-Path` |
| `CUDA out of memory` | `batch=8` + `imgsz=832` too large for GPU | Reduce batch: edit `scripts/train.py:53` to `batch=4` or train `yolo11n`; or lower `imgsz=640` |
| Training extremely slow | Running on CPU | Install CUDA PyTorch + pass `--device 0`; or use `yolo11n` |
| `dataset/data.yaml` names mismatch | (Fixed) Both configs now use English names | No action needed — either `config/data.yaml` or `dataset/data.yaml` works |
| `ModuleNotFoundError: ultralytics` | Venv not activated | Re-run `.\.venv\Scripts\Activate.ps1` and `pip install -r requirements.txt` |
| Mosaic/augmentation looks wrong | Stretch-resized 512 dataset + `imgsz=832` | This is intentional — Ultralytics re-augments at train time. For strict reproducibility use `--variant yolo11s_preprocessing` which pins aug params |

**Quick health check:**

```powershell
python scripts/validate_repository.py
# Requires docs/*.md — if missing, ignore and focus on training files only
```

---

## 8. Reproducibility Notes

- **Recorded device:** NVIDIA Tesla T4 (see `config\experiments.yaml:6`)
- **Hyperparameters:** Pinned in `config\experiments.yaml` + `scripts\train.py:11` (`VARIANTS` dict)
- **Image size:** `832` (different from export `512` — Ultralytics resizes on the fly)
- **Epochs:** `150` with early stopping; reference completed epochs logged in `experiments.yaml`
- **To exactly reproduce thesis run:** use `--variant yolo11s_preprocessing` which logs all aug params

---

## 9. Responsible Use Reminder

This is an academic research prototype for contextual scene review with **human-in-the-loop**. Do not use for identity recognition or automated punitive decisions. Validate locally, document errors/bias, and keep human review mandatory — see `README.md:34`.

---

## 10. Quick Command Cheatsheet

```powershell
# Setup (once)
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt

# Fix config/data.yaml → path: dataset   (edit file)

# Train
python scripts/train.py --variant yolo11s --data config/data.yaml --device 0

# Inference
python scripts/inference.py --model runs/train/yolo11s/weights/best.pt --source dataset/test/images --confidence 0.35

# Validate
yolo val model=runs/train/yolo11s/weights/best.pt data=config/data.yaml imgsz=832
```
