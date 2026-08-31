# Contextual Scene Review

A research prototype built on **YOLO11** for contextual scene analysis in public-area imagery. The model detects and localizes individuals, labeling them as either `person` or `person_with_object` to support **human-in-the-loop review workflows**.

---

## Overview

This repository demonstrates a **research-focused system** for reviewing public-space scenes.  
Key design principles:  
- Prioritizes **contextual understanding** over threat detection.  
- Avoids identity recognition and automated decision-making.  
- Ensures **human oversight** remains central to the workflow.  

---

## Labels

The model outputs two simple categories:  
- `person`  
- `person_with_object`  

---

## Dataset

- Dataset follows **YOLO format** with train/valid/test splits.  
- Located in: `data/dataset`  
- Original dataset available via [Google Drive](https://drive.google.com/file/d/1rLZcHEgoo1Y3S8jL9LBE2pGXdc3-ey5G/view?usp=sharing).  
- Review dataset carefully before reuse to ensure suitability for your application.  

---

## Quick Start

Set up environment and run inference:

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/inference.py --model path/to/best.pt --source path/to/image-or-video --confidence 0.35
```

---

## Training

Train a YOLO11 variant with your dataset:

```bash
python scripts/train.py --variant yolo11s --data config/data.yaml
```

---

## Responsible Use

- **Human review is mandatory** — this system is not autonomous.  
- Do **not** use for identity recognition or punitive automation.  
- Validate thoroughly before deployment.  
- Document errors, bias, and edge cases to ensure transparency.  

---

## License

Distributed under the **GNU 3.0** license.  

---