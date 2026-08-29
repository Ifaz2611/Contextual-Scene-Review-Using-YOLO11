# Contextual Scene Review

A YOLO11-based research project for contextual scene review in public-area imagery. The model localizes people and labels them as `person` or `person_with_object` for human review workflows.

## Overview

This repository demonstrates a research prototype for reviewing public-space scenes with a human-in-the-loop. It focuses on contextual understanding rather than threat classification or identity inference.

## Labels

- `person`
- `person_with_object`

## Dataset

The project uses a YOLO-format dataset in `data/dataset` with train/valid/test splits. The original dataset is available from the linked Google Drive folder and should be reviewed before reuse.

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/inference.py --model path/to/best.pt --source path/to/image-or-video --confidence 0.35
```

## Training

```bash
python scripts/train.py --variant yolo11s --data config/data.yaml
```

## Responsible use

- Human review remains required.
- Do not use the model for identity recognition or automated punitive decisions.
- Validate locally before deployment and document errors, bias, and edge cases.

## License

This project is distributed under the AGPL-3.0 license.
