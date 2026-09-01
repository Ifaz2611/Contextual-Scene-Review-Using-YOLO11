#!/usr/bin/env python3
"""CPU demo for Ryzen 7 5800H + Radeon (no NVIDIA) - inference + tiny train."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# only for fine-tuned 2-class weights
ENGLISH_LABELS = {0: "person", 1: "person_with_Dangerous_object"}

def predict_cpu(model_path: str = "yolo11n.pt", source: str = "dataset/test/images", conf: float = 0.35):
    model = YOLO(model_path)
    # only override names if model actually has 2 classes (finetuned)
    nc = getattr(model.model, "names", {})
    is_custom = isinstance(nc, dict) and len(nc) == 2
    if is_custom:
        model.model.names = ENGLISH_LABELS
        print(f"[info] using custom 2-class labels {ENGLISH_LABELS}")
    else:
        print(f"[info] using COCO pretrained labels (nc={len(nc)}), not overriding")

    print(f"[demo] predicting with {model_path} on {source} (device=cpu, imgsz=640)")
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        project="runs/predict",
        name="demo_cpu",
        exist_ok=True,
        device="cpu",
        imgsz=640,
        verbose=True,
    )
    print(f"Done -> runs/predict/demo_cpu ( {len(results)} images )")
    # print first result summary
    for r in results[:3]:
        print(r.verbose())
    return 0

def train_demo(data: str = "config/data.yaml", epochs: int = 2):
    """Tiny training that actually fits in 15.4GB RAM on CPU in ~15-30 min."""
    model = YOLO("yolo11n.pt")  # smallest, fastest for CPU
    print(f"[demo] tiny train epochs={epochs} data={data} device=cpu batch=4 imgsz=640")
    model.train(
        data=data,
        epochs=epochs,
        batch=4,          # 8 OOM on 15GB on CPU for 832, use 4
        imgsz=640,        # 832 slow, 640 is 40% faster on CPU
        device="cpu",
        workers=4,        # 8 too high for Windows, 4 safe
        project="runs/train",
        name="demo_cpu_train",
        patience=10,
        exist_ok=True,
    )
    print("Tiny train done -> runs/train/demo_cpu_train/weights/best.pt")
    print("Now run: python scripts/demo_cpu.py --mode predict --model runs/train/demo_cpu_train/weights/best.pt")
    return 0

def parse_args():
    p = argparse.ArgumentParser(description="CPU demo (Ryzen 7 5800H, no CUDA)")
    p.add_argument("--mode", choices=["predict", "train"], default="predict", help="predict uses pretrained yolo11n.pt, train does 2 epochs on yolo11n")
    p.add_argument("--model", default="yolo11n.pt", help="weights for predict")
    p.add_argument("--source", default="dataset/test/images", help="image/video/dir/0 for webcam")
    p.add_argument("--data", default="config/data.yaml", help="data yaml for train")
    p.add_argument("--epochs", type=int, default=2, help="epochs for train demo")
    p.add_argument("--conf", type=float, default=0.35)
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "predict":
        return predict_cpu(args.model, args.source, args.conf)
    else:
        return train_demo(args.data, args.epochs)

if __name__ == "__main__":
    raise SystemExit(main())
