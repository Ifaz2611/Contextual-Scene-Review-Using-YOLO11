#!/usr/bin/env python3
"""
Overall runner for Contextual-Scene-Review YOLO11 (mini PC CPU).
One file to setup-check -> pretrained demo -> train -> predict with best.pt -> validate.

Usage (PowerShell, project root):
  .\\.venv\\Scripts\\Activate.ps1
  python run_all.py                 # fast: pretrained demo only (~1 min)
  python run_all.py --train --epochs 20  # + mini-PC training (2-5h, accurate)
  python run_all.py --train --epochs 2   # + smoke train (15-30m, inaccurate - demo only)
  python run_all.py --skip-pretrained --train --epochs 20  # only train+predict
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# reuse fixed logic from scripts/demo_cpu.py
from scripts.demo_cpu import predict_cpu, train_demo, ENGLISH_LABELS
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = "config/data.yaml"
TEST_SRC = "dataset/test/images"
PRETRAINED = "yolo11n.pt"

def check_env():
    print("="*70)
    print("[1/5] Checking environment...")
    ok = True
    if not Path(DATA).is_file():
        print(f"  [FAIL] {DATA} not found")
        ok = False
    else:
        print(f"  [OK] {DATA}")
        print("  " + Path(DATA).read_text().strip().replace("\n","\n  "))
    for p in [f"dataset/train/images", f"dataset/valid/images", f"dataset/test/images"]:
        exists = Path(p).exists()
        count = len(list(Path(p).glob("*.jpg"))) if exists else 0
        print(f"  {'[OK]' if exists else '[FAIL]'} {p} -> {count} images")
        if not exists: ok=False
    # check imports
    try:
        import ultralytics, torch, yaml, PIL
        print(f"  [OK] ultralytics {ultralytics.__version__}, torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    except Exception as e:
        print(f"  [FAIL] imports: {e}")
        ok=False
    if not ok:
        print("\nFix: pip install -r requirements.txt and run from project root")
        sys.exit(1)
    print("  -> all checks passed\n")

def run_pretrained_demo(conf=0.35, imgsz=640):
    print("="*70)
    print(f"[2/5] Pretrained demo (COCO yolo11n.pt) -> {TEST_SRC}")
    print("  Note: COCO can't distinguish person_with_Dangerous_object, expect [WARN]")
    predict_cpu(model_path=PRETRAINED, source=TEST_SRC, conf=conf, imgsz=imgsz)
    print()

def run_train(epochs, imgsz=640):
    print("="*70)
    print(f"[3/5] Training demo (epochs={epochs})...")
    if epochs <= 5:
        print("  WARNING: 2-5 epochs is SMOKE TEST only, accuracy will be poor!")
    train_demo(data=DATA, epochs=epochs, imgsz=imgsz)
    print()

def run_predict_best(conf=0.35, imgsz=640):
    print("="*70)
    print(f"[4/5] Predict with trained best.pt (conf={conf})")
    best = Path("runs/train/demo_cpu_train/weights/best.pt")
    if not best.is_file():
        # fallback to full train variants
        for cand in [Path("runs/train/yolo11s/weights/best.pt"), Path("runs/train/yolo11n/weights/best.pt")]:
            if cand.is_file():
                best = cand
                break
    if not best.is_file():
        print(f"  [SKIP] no best.pt found at {best}")
        print(f"  Train first: python run_all.py --train --epochs 20")
        return
    print(f"  Using {best}")
    predict_cpu(model_path=str(best), source=TEST_SRC, conf=conf, imgsz=imgsz)
    print()

def run_validate(imgsz=640):
    print("="*70)
    print(f"[5/5] Validation (mAP) on val split")
    best = Path("runs/train/demo_cpu_train/weights/best.pt")
    if not best.is_file():
        print(f"  [SKIP] {best} not found, skipping yolo val")
        print(f"  To validate pretrained: yolo val model=yolo11n.pt data={DATA} imgsz={imgsz}")
        return
    try:
        model = YOLO(str(best))
        if len(getattr(model.model, "names", {})) == 2:
            model.model.names = ENGLISH_LABELS
        metrics = model.val(data=DATA, imgsz=imgsz, device="cpu", verbose=True)
        print(f"  Metrics: {metrics}")
        try:
            print(f"  mAP50-95: {metrics.box.map:.3f}  mAP50: {metrics.box.map50:.3f}")
        except: pass
    except Exception as e:
        print(f"  [FAIL] val: {e}")
    print()
    print("Results:")
    print("  Images -> runs/predict/demo_cpu/")
    print("  Weights -> runs/train/demo_cpu_train/weights/best.pt")
    print("  Check: Get-ChildItem -Recurse runs | Select-Object FullName")

def parse_args():
    p = argparse.ArgumentParser(description="Overall runner for YOLO11 mini-PC")
    p.add_argument("--train", action="store_true", help="also train (default: pretrained demo only)")
    p.add_argument("--epochs", type=int, default=20, help="epochs if --train (2=smoke 15-30=usable, 150=full)")
    p.add_argument("--conf", type=float, default=0.35, help="confidence 0.0-1.0")
    p.add_argument("--imgsz", type=int, default=640, help="640 for mini-PC, 832 for full")
    p.add_argument("--skip-pretrained", action="store_true", help="skip pretrained demo")
    return p.parse_args()

def main():
    args = parse_args()
    check_env()
    if not args.skip_pretrained:
        run_pretrained_demo(conf=args.conf, imgsz=args.imgsz)
    else:
        print("[2/5] Skipped pretrained demo\n")
    if args.train:
        run_train(epochs=args.epochs, imgsz=args.imgsz)
        run_predict_best(conf=args.conf, imgsz=args.imgsz)
        run_validate(imgsz=args.imgsz)
    else:
        print("="*70)
        print("[3/5] Skipped training (use --train to enable)")
        print("  For accurate 2-class detection you need:")
        print("    python run_all.py --train --epochs 20")
        print("  Then predict uses runs/train/demo_cpu_train/weights/best.pt")
        print()
        print("Next steps:")
        print("  python run_all.py --train --epochs 20   # 2-5h, accurate")
        print("  python scripts/train.py --variant yolo11s --data config/data.yaml --device cpu # full")
    print("\nDone.")

if __name__ == "__main__":
    raise SystemExit(main())
