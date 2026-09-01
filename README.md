# Contextual Scene Review — YOLO11

Research prototype on **YOLO11** for contextual scene analysis. Detects `person` vs `person_with_object` to support **human-in-the-loop** review — no identity recognition, no automated decisions.

Verified on **Windows 11, AMD Ryzen 7 5800H (8C/16T), 15.4GB RAM, Radeon Graphics (no NVIDIA), torch 2.13.0+cpu, Python 3.14.5** at ~63ms/image `640x640` CPU.

---

## 1. Dataset

Bundled, YOLO format:

```
dataset/
├── data.yaml          # also valid, but use config/data.yaml for training
├── train/images/ 2512 + train/labels/
├── valid/images/ 314  + valid/labels/
└── test/images/ 314   + test/labels/   <- demo runs on this
```

Classes `config/data.yaml:6` :
- `0: person`
- `1: person_with_object`

`config/data.yaml:1` is fixed to `path: dataset` (was `../data/dataset`). If you move the project, keep `path: dataset` when running from project root, or use absolute `C:/.../dataset`.

Source: Roboflow export, `512x512` stretch, labels normalized `class cx cy w h`.

---

## 2. Setup

```powershell
python --version  # 3.10-3.12 recommended, 3.14 works
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  # if blocked
python -m pip install --upgrade pip
python -m pip install -r requirements.txt  # ultralytics>=8.3, PyYAML, Pillow -> pulls torch CPU
python -c "from ultralytics import YOLO; print(YOLO)"
python -c "import torch; print(torch.cuda.is_available())"  # False on this PC -> CPU
```

No NVIDIA needed. `yolo11n.pt` (~5MB) auto-downloads, `yolo11s.pt` (19MB) already included.

---

## 3. Quick Demo (CPU, this device)

### 3.1 Inference — pretrained (no training needed)

```powershell
python scripts/demo_cpu.py --mode predict --model yolo11n.pt --source dataset/test/images --conf 0.35
# also:
# --source path/to/image.jpg --source path/to/video.mp4 --source 0  # webcam
# alternative (fixed for COCO vs 2-class):
# python scripts/inference.py --model yolo11n.pt --source dataset/test/images --confidence 0.35 --device cpu
```

Proven run: 314 images, `640x640`, 63.7ms inference on Ryzen 7 5800H CPU.

### 3.2 Tiny train — 2 epochs, fits 15GB RAM

```powershell
python scripts/demo_cpu.py --mode train --epochs 2 --data config/data.yaml
# equivalent to: YOLO("yolo11n.pt").train(data="config/data.yaml", epochs=2, batch=4, imgsz=640, device="cpu", workers=4)
# then:
python scripts/demo_cpu.py --mode predict --model runs/train/demo_cpu_train/weights/best.pt --source dataset/test/images
```

Full training (days on CPU, use GPU if available):

```powershell
python scripts/train.py --variant yolo11n --data config/data.yaml --device cpu  # or --device 0 for CUDA
python scripts/train.py --variant yolo11s --data config/data.yaml --device cpu
# variants: yolo11n / yolo11s / yolo11s_preprocessing / yolo11m / yolo11l  (config/experiments.yaml:8)
```

---

## 4. Where to See Results

### Inference results (annotated images)

Ultralytics 8.4 on this repo nests under `runs/detect/` :

```powershell
# actual verified path after demo:
explorer runs\detect\runs\predict\demo_cpu
# expected generic path (some versions):
explorer runs\predict\demo_cpu
# list:
Get-ChildItem -Recurse runs | Select-Object FullName
```

Each run contains 314 `.jpg` with boxes + `labels/` txt if `save_txt=True`.

### Training results

```powershell
explorer runs\train                    # or runs\detect\runs\train on this install
# per run e.g.:
runs/train/demo_cpu_train/
├── weights/best.pt      # <-- use this for inference
├── weights/last.pt
├── args.yaml
├── results.csv
├── confusion_matrix.png
├── F1_curve.png / PR_curve.png / P_curve.png / R_curve.png
└── train_batch*.jpg
# also for full variants:
runs/train/yolo11s/weights/best.pt
runs/train/yolo11n/weights/best.pt
```

Validate:

```powershell
yolo val model=runs/train/demo_cpu_train/weights/best.pt data=config/data.yaml imgsz=640
yolo val model=yolo11n.pt data=config/data.yaml imgsz=640  # pretrained baseline
```

---

## 5. Project Structure

```
.
├── config/data.yaml        # train/val/test paths, nc=2
├── config/experiments.yaml # variants + patience + recorded Tesla T4
├── dataset/                # bundled data
├── scripts/train.py        # VARIANTS dict, epochs=150 batch=8 imgsz=832
├── scripts/inference.py    # --model --source --confidence --project --name --device
├── scripts/demo_cpu.py     # CPU demo (this PC): predict/train modes, imgsz=640 batch=4
├── requirements.txt
├── GET_STARTED.md          # detailed guide + troubleshooting
└── runs/                   # outputs (gitignored)
```

---

## 6. Troubleshooting

| Issue | Fix |
|---|---|
| `Dataset ... images not found` | Ensure `config/data.yaml:1` = `path: dataset` and run from project root. Verify `Test-Path dataset\train\images` |
| `KeyError: 27` on inference | Fixed in `scripts/inference.py:40` — only overrides names for 2-class weights |
| `CUDA out of memory` / slow CPU | Use `yolo11n`, `batch=4`, `imgsz=640`, `workers=4` (`scripts/demo_cpu.py:48-53`) |
| `ModuleNotFoundError: ultralytics` | `.\.venv\Scripts\Activate.ps1` + `pip install -r requirements.txt` |

---

## 7. Responsible Use

Human review mandatory. Do not use for identity recognition or punitive automation. Validate, document errors/bias, keep human oversight — see `GET_STARTED.md:9`.

## License

GNU GPL 3.0
