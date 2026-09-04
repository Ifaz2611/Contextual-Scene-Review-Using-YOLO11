#!/usr/bin/env python3
"""Run YOLO11 inference with clear English display labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ENGLISH_LABELS = {
    0: "person",
    1: "person_with_Dangerous_object",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained two-class YOLO11 detector."
    )
    parser.add_argument("--model", required=True, help="Path to a trained .pt file.")
    parser.add_argument("--source", required=True, help="Image, video, directory, or stream source.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold in [0, 1] (try 0.25 for weak/2-epoch models).")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (640 for mini-PC/demo, 832 to match full training).")
    parser.add_argument("--project", default="runs/predict", help="Output parent directory.")
    parser.add_argument("--name", default="demo", help="Output run name.")
    parser.add_argument("--device", default=None, help="Ultralytics device value, such as cpu, 0, or 0,1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(str(model_path))
    names = getattr(model.model, "names", {})
    # only override for 2-class fine-tuned weights, COCO pretrained has 80 classes
    if isinstance(names, dict) and len(names) == 2:
        model.model.names = ENGLISH_LABELS
        print(f"[info] 2-class fine-tuned weights detected — using {ENGLISH_LABELS}")
    else:
        n = len(names) if isinstance(names, dict) else 0
        print(f"[WARN] model has nc={n} (not 2). If you trained demo 2 epochs,")
        print("       detection WILL be weak. For accurate results use:")
        print("         python scripts/train.py --variant yolo11s --data config/data.yaml --device cpu")
        print("       then inference with runs/train/<variant>/weights/best.pt")
        if n == 80:
            print("       COCO pretrained (yolo11n.pt/yolo11s.pt) cannot distinguish")
            print("       'person_with_Dangerous_object' — it only knows generic 'person'.")

    # if conf is high and model is weak, suggest lower
    if args.confidence > 0.5:
        print(f"[hint] conf={args.confidence} is high — weak models may need --confidence 0.25-0.35")

    print(f"[info] predict source={args.source} imgsz={args.imgsz} conf={args.confidence} iou={args.iou} device={args.device}")

    results = model.predict(
        source=args.source,
        conf=args.confidence,
        iou=args.iou,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=args.device,
        imgsz=args.imgsz,
        verbose=True,
    )
    print(f"Saved predictions to {Path(args.project) / args.name} ({len(results)} images)")
    # hint if no detections
    empty = sum(1 for r in results if len(r.boxes) == 0)
    if empty:
        print(f"[hint] {empty}/{len(results)} images had 0 boxes — try --confidence 0.25 or verify you used best.pt")
    for r in results[:2]:
        print(r.verbose())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
