# Getting Started: Train YOLO11 Easily

This project trains a YOLO11 object-detection model to find:

- `person`
- `person_with_Dangerous_object`

The dataset is already included in the `dataset\` folder, so you can start
training without downloading or preparing another dataset.

> **Important:** This is a research prototype. Model predictions must be
> checked by a human. Do not use it for identity recognition or automatic
> punishment.

## How the project works

The training process is:

```text
Images + labels
        |
        v
config\data.yaml
        |
        v
YOLO11 learns the two classes
        |
        v
runs\train\...\weights\best.pt
        |
        v
Inference on new images, videos, or a webcam
```

### What each important folder does

| Path | Purpose |
|---|---|
| `dataset\train\` | Images and labels used to learn |
| `dataset\valid\` | Images used to check progress during training |
| `dataset\test\` | Images used for final examples and testing |
| `config\data.yaml` | Dataset location, splits, and class names |
| `scripts\train.py` | Full YOLO11 training script |
| `scripts\demo_cpu.py` | Small, CPU-friendly training and prediction demo |
| `scripts\inference.py` | Prediction using a trained model |
| `run_all.py` | One-command setup check, training, prediction, and validation |
| `runs\` | Generated weights, charts, and prediction results |

Each image has a matching text file in the `labels` folder. A label line uses
the YOLO format:

```text
class_id center_x center_y width height
```

The coordinates are normalized between `0` and `1`.

## 1. Open PowerShell in the project folder

Run all commands from the repository root, the folder containing
`run_all.py`, `dataset\`, and `requirements.txt`.

```powershell
cd "C:\Users\Ifaz md zahin\Downloads\train\Contextual-Scene-Review-Using-YOLO11"
```

Check that the dataset exists:

```powershell
Test-Path "dataset\train\images"
Test-Path "dataset\valid\images"
Test-Path "dataset\test\images"
```

Each command should print `True`.

## 2. Install the project

Create and activate a virtual environment once:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the same PowerShell window and
then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You can confirm that the installation works:

```powershell
python -c "from ultralytics import YOLO; print('Ultralytics is ready')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

NVIDIA CUDA is optional. CPU training works, but it is much slower.

## 3. Easiest option: run the complete workflow

The easiest way to try the project is:

```powershell
.\.venv\Scripts\Activate.ps1
python run_all.py --train --epochs 20
```

This command:

1. Checks the environment and dataset.
2. Runs a pretrained YOLO11 demo.
3. Trains a small CPU-friendly YOLO11 model for 20 epochs.
4. Predicts on `dataset\test\images`.
5. Validates the trained model and prints metrics.

Twenty epochs is a practical starting point for a CPU computer. It is not the
same as the full research training run, but it should produce a useful model.

### Choose the training length

```powershell
# Very quick smoke test. The predictions will be inaccurate.
python run_all.py --train --epochs 2

# Recommended CPU starting point.
python run_all.py --train --epochs 20

# More training for better accuracy, but it takes longer.
python run_all.py --train --epochs 50
```

The 2-epoch command only checks that the code works. Do not use its model for
reliable results.

## 4. Full training with `scripts\train.py`

For the standard training configuration, use YOLO11s:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\train.py --variant yolo11s --data config/data.yaml --device cpu
```

If you have an NVIDIA GPU with CUDA installed, use:

```powershell
python scripts\train.py --variant yolo11s --data config/data.yaml --device 0
```

The full configuration trains for up to 150 epochs and uses early stopping.
It may take hours or days on a CPU. Start with `run_all.py --train --epochs
20` before starting a long run.

Available model variants:

| Variant | Speed | Accuracy potential | Recommended use |
|---|---:|---:|---|
| `yolo11n` | Fastest | Lowest | CPU testing |
| `yolo11s` | Fast | Good | Recommended default |
| `yolo11s_preprocessing` | Fast | Good | Reproduce recorded augmentation |
| `yolo11m` | Slower | Higher | GPU training |
| `yolo11l` | Slowest | Highest potential | Powerful GPU |

For example:

```powershell
python scripts/train.py --variant yolo11n --data config/data.yaml --device cpu
python scripts/train.py --variant yolo11s_preprocessing --data config/data.yaml --device 0
```

## 5. Find the trained model

The most important output is:

```text
runs\train\demo_cpu_train\weights\best.pt
```

For a full variant, it is usually:

```text
runs\train\yolo11s\weights\best.pt
```

`best.pt` is the checkpoint with the best validation performance. Use it for
prediction instead of `last.pt`.

List all generated files with:

```powershell
Get-ChildItem -Recurse runs | Select-Object FullName
```

Training charts such as `results.csv`, `confusion_matrix.png`, and precision/
recall curves are saved inside the training folder.

## 6. Run the trained model on new data

After training, run prediction on the test images:

```powershell
python scripts/inference.py `
  --model "runs\train\demo_cpu_train\weights\best.pt" `
  --source "dataset\test\images" `
  --confidence 0.35 `
  --device cpu
```

You can replace `--source` with an image, video, folder, or webcam:

```powershell
# One image
python scripts/inference.py --model "runs\train\demo_cpu_train\weights\best.pt" --source "photo.jpg"

# A video
python scripts/inference.py --model "runs\train\demo_cpu_train\weights\best.pt" --source "video.mp4"

# Webcam
python scripts/inference.py --model "runs\train\demo_cpu_train\weights\best.pt" --source 0 --device cpu
```

Predictions are saved in:

```text
runs\predict\demo\
```

If there are too few detections, try a lower confidence threshold:

```powershell
--confidence 0.25
```

## 7. Validate the model

To calculate validation metrics for a trained model:

```powershell
yolo val `
  model="runs\train\demo_cpu_train\weights\best.pt" `
  data="config\data.yaml" `
  imgsz=640 `
  device=cpu
```

For a full YOLO11s run, change the model path to:

```text
runs\train\yolo11s\weights\best.pt
```

Important metrics include `mAP50`, `mAP50-95`, precision, and recall. A metric
alone does not prove that the model is safe or suitable for deployment; inspect
the annotated images and false detections too.

## Common problems

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: ultralytics` | Activate `.venv` and run `python -m pip install -r requirements.txt` |
| Dataset not found | Run from the project root and confirm `config\data.yaml` contains `path: dataset` |
| PowerShell refuses activation | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Training is too slow | Use `yolo11n`, fewer epochs, or an NVIDIA GPU |
| CUDA out of memory | Use `--device cpu`, use `yolo11n`, or reduce batch/image size in `scripts\train.py` |
| No prediction boxes | Use the trained `best.pt`, not the pretrained COCO weights, and try `--confidence 0.25` |
| Results are poor after 2 epochs | Expected: 2 epochs is only a smoke test; train for 20 or more epochs |

## Quick copy-paste commands

```powershell
cd "C:\Users\Ifaz md zahin\Downloads\train\Contextual-Scene-Review-Using-YOLO11"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_all.py --train --epochs 20
```

After the command finishes, inspect the images in
`runs\predict\demo_cpu\` and use
`runs\train\demo_cpu_train\weights\best.pt` for future predictions.
