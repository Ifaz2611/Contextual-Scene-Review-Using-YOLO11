from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from ultralytics import YOLO

# must match config/data.yaml:7-8 — keep in sync!
ENGLISH_LABELS = {0: "person", 1: "person_with_Dangerous_object"}


def predict_cpu(model_path: str = "yolo11n.pt", source: str = "dataset/test/images", conf: float = 0.35, imgsz: int = 640):
    model = YOLO(model_path)
    nc = getattr(model.model, "names", {})
    is_custom = isinstance(nc, dict) and len(nc) == 2
    if is_custom:
        model.model.names = ENGLISH_LABELS
        print(f"[info] using custom 2-class labels {ENGLISH_LABELS}")
    else:
        print(f"[WARN] using COCO pretrained labels (nc={len(nc)}), not 2-class fine-tuned.")
        print("       -> This will detect generic 'person' but CANNOT distinguish")
        print("          'person_with_Dangerous_object'. Accuracy will appear low.")
        print("       -> To get 2-class accuracy, train first then use:")
        print("          runs/train/demo_cpu_train/weights/best.pt")

    # auto-lower conf slightly for weak/early models to show more detections
    print(f"[demo] predicting with {model_path} on {source} (device=cpu, imgsz={imgsz}, conf={conf})")
    results = model.predict(
        source=source,
        conf=conf,
        iou=0.5,
        save=True,
        project="runs/predict",
        name="demo_cpu",
        exist_ok=True,
        device="cpu",
        imgsz=imgsz,
        verbose=True,
    )
    print(f"Done -> runs/predict/demo_cpu ({len(results)} images)")
    # print first result summary
    for r in results[:3]:
        print(r.verbose())
        if len(r.boxes) == 0:
            print("  [hint] no boxes — try --conf 0.25 or check you are using best.pt not yolo11n.pt")
    return 0


def train_demo(data: str = "config/data.yaml", epochs: int = 2, imgsz: int = 640):
    """Mini-PC friendly training. 2 epochs is a smoke-test only — not accurate!

    Accuracy guidance:
      epochs=2   -> smoke test, ~15-30min, mAP very low, boxes inaccurate (expected)
      epochs=15-30 -> usable mini-PC preset, ~2-5h on Ryzen 7 5800H CPU, much more accurate
      epochs=50-150 -> full training via scripts/train.py (use GPU if available)
    """
    # --- prominent warning for inaccurate regime ---
    if epochs <= 5:
        print("=" * 70)
        print(f"[WARN] epochs={epochs} is a SMOKE TEST only — detection WILL be inaccurate!")
        print("       The model barely learns in 2-5 epochs. Expect poor boxes /")
        print("       many misses and confusion between 'person' vs")
        print("       'person_with_Dangerous_object'.")
        print("       For usable accuracy on your mini PC, re-run with:")
        print("         python scripts/demo_cpu.py --mode train --epochs 20")
        print("       For best accuracy (hours/days on CPU, use GPU if possible):")
        print("         python scripts/train.py --variant yolo11s --data config/data.yaml")
        print("=" * 70)
    elif epochs < 15:
        print(f"[info] epochs={epochs} is better than 2 but still low. For usable")
        print("       accuracy on mini PC, prefer --epochs 20-30.")

    # Safe workers for mini PC / Windows: cap at cpu_count // 2, max 4, min 0
    try:
        cpu_n = multiprocessing.cpu_count()
    except Exception:
        cpu_n = 4
    workers = max(0, min(4, cpu_n // 2))
    # ultralytics on Windows with workers>0 can be unstable on some mini PCs
    # keep 2 as safe default if 4 causes hangs
    if workers > 2:
        workers = 2
    print(f"[info] auto workers={workers} (cpu_count={cpu_n})")

    model = YOLO("yolo11n.pt")  # smallest, fastest for CPU
    print(f"[demo] mini-PC train epochs={epochs} data={data} device=cpu batch=4 imgsz={imgsz} workers={workers}")

    # stronger augmentation than before — helps generalization even for short runs
    # keeps batch=4 and imgsz=640 safe for 15GB RAM / mini PC
    model.train(
        data=data,
        epochs=epochs,
        batch=4,
        imgsz=imgsz,
        device="cpu",
        workers=workers,
        project="runs/train",
        name="demo_cpu_train",
        patience=15 if epochs > 10 else 10,
        exist_ok=True,
        # augmentation (mirrors yolo11s_preprocessing but lighter for CPU)
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        mosaic=0.7,
        mixup=0.05,
        close_mosaic=10,
        # less aggressive cache to save RAM on mini PC
        cache=False,
        plots=True,
        val=True,
    )
    print("Tiny train done -> runs/train/demo_cpu_train/weights/best.pt")

    # --- post-train validation so user sees actual mAP ---
    try:
        print("\n[demo] validating best.pt on val split...")
        best = Path("runs/train/demo_cpu_train/weights/best.pt")
        if best.is_file():
            val_model = YOLO(str(best))
            # ensure names are correct for val display
            if len(getattr(val_model.model, "names", {})) == 2:
                val_model.model.names = ENGLISH_LABELS
            metrics = val_model.val(data=data, imgsz=imgsz, device="cpu", verbose=True)
            print(f"[demo] val metrics: {metrics}")
            # crude accuracy hint
            try:
                map50 = float(getattr(metrics.box, "map50", 0) or 0)
                if map50 < 0.3:
                    print(f"[WARN] mAP50={map50:.3f} is low — model is under-trained.")
                    print("       Increase --epochs to 20-30 and re-train, or use")
                    print("       scripts/train.py --variant yolo11s for full training.")
                else:
                    print(f"[info] mAP50={map50:.3f} — reasonable for {epochs} epochs.")
            except Exception:
                pass
        else:
            print(f"[WARN] best.pt not found at {best}")
    except Exception as e:
        print(f"[WARN] validation failed: {e}")

    print("\nNow run:")
    print("  python scripts/demo_cpu.py --mode predict --model runs/train/demo_cpu_train/weights/best.pt --source dataset/test/images --conf 0.35")
    print("  # if no boxes: try --conf 0.25")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="CPU demo (Ryzen 7 5800H / mini PC, no CUDA)")
    p.add_argument("--mode", choices=["predict", "train"], default="predict", help="predict uses weights, train does mini-PC training")
    p.add_argument("--model", default="yolo11n.pt", help="weights for predict")
    p.add_argument("--source", default="dataset/test/images", help="image/video/dir/0 for webcam")
    p.add_argument("--data", default="config/data.yaml", help="data yaml for train")
    p.add_argument("--epochs", type=int, default=2, help="epochs for train demo (2=smoke test, 20=usable mini-PC, 150=full)")
    p.add_argument("--conf", type=float, default=0.35, help="confidence threshold 0.0-1.0 (try 0.25 for weak models)")
    p.add_argument("--imgsz", type=int, default=640, help="inference/train image size (640 for mini PC, 832 for full train)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "predict":
        return predict_cpu(args.model, args.source, args.conf, args.imgsz)
    else:
        return train_demo(args.data, args.epochs, args.imgsz)


if __name__ == "__main__":
    raise SystemExit(main())
