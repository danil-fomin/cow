# YOLO classification training — design

**Date:** 2026-06-08
**Branch:** `yolo-cls`
**Status:** approved (design), pending implementation plan

## Goal

Replace the project's custom BCS classifier (EfficientNet/YOLO backbone + CBAM +
ConvHead + CORN head, custom training loop, DDP) with **native Ultralytics YOLO
classification training** (`yolo26n-cls`). The branch is dedicated to this
approach; the custom pipeline stays on `main`.

The task: predict cow Body Condition Score (BCS), an **ordinal** target with
values `[3.25, 3.5, 3.75, 4.0, 4.25]`.

## Decisions (settled during brainstorming)

1. **Training engine:** native Ultralytics `model.train(...)`. No custom
   `nn.Module`, no custom training loop, no manual DDP/optimizer/scheduler.
2. **Scope:** full replacement on this branch — delete the custom NN code, loss,
   loop, dataset/transforms, and DDP/checkpoint helpers.
3. **Metrics:** keep the ordinal BCS metrics (QWK, MAE, adjacent-accuracy)
   computed **post-hoc** via the existing `src/metrics.py:compute_metrics`, in
   addition to Ultralytics' native top-1 accuracy.
4. **Data layout:** produce exactly the layout Ultralytics classification
   expects — no assumptions about class-folder naming imposed by us.

## Architecture

Thin orchestration around Ultralytics:

```
dataset/dataset/<class>/*.{jpg,...}        (source, class-foldered)
        │  prepare_dataset.py  (seeded per-class split, preserve class names)
        ▼
dataset_cls/{train,val,test}/<class>/*.jpg (Ultralytics classification layout)
        │  train.py  → YOLO("yolo26n-cls.pt").train(data="dataset_cls", ...)
        ▼
runs/.../weights/best.pt
        │  evaluate.py → model.val()  +  post-hoc ordinal metrics over test
        ▼
metrics report + confusion_matrix.png
        │  predict.py → model.predict(source) → top-1 idx → class_values[idx]
        ▼
predictions
```

## Components

### 1. Data preparation — rewrite `src/data/prepare_dataset.py`
- **Input:** `dataset/dataset/<class>/*.{jpg,jpeg,png,...}` (already foldered by
  class).
- **Output:** `dataset_cls/{train,val,test}/<class>/<file>` — the exact layout
  Ultralytics classification consumes.
- **Class folder names: preserved verbatim from the source.** We do not rename to
  indices or values. Ultralytics derives class indices by sorting these names.
- Reuse the existing seeded, per-class proportional split logic (`splits` ratios,
  `seed=42`, sorted-then-shuffled file order) so splits are reproducible.
- `class_values` is **not** used for naming here; it is only the semantic
  value table used later for ordinal metrics and predict output.

### 2. Training — rewrite `src/training/train.py`
A thin wrapper:
```python
model = YOLO(config["model"])              # "yolo26n-cls.pt"
model.train(
    data="dataset_cls",
    epochs=config["epochs"],
    imgsz=config["img_size"],
    batch=config["batch_size"],
    lr0=float(config["learn_rate"]),
    weight_decay=float(config["weight_decay"]),
    patience=config["early_stopping_patience"],
    device=<derived from num_gpus>,        # e.g. "0,1" for multi-GPU DDP
)
```
Ultralytics handles augmentation, optimizer, scheduler, DDP, checkpointing
(`best.pt`/`last.pt`), early stopping, and training curves internally.

### 3. Evaluation — rewrite `src/training/runner.py:run_evaluate` + trim `src/evaluate.py`
- Run `model.val(...)` for native Ultralytics metrics.
- **Post-hoc ordinal metrics:** run the model over the `test` split, collect
  `(pred_index, target_index)`, derive the **index→BCS-value mapping from
  `model.names`** (Ultralytics' sorted class names) against `class_values`, then
  feed indices to the existing `compute_metrics` → QWK / MAE / adjacent-accuracy.
- Keep the confusion-matrix rendering from `evaluate.py`. Drop
  `collect_predictions` (old-loop coupled) and `plot_training_curves` (curves now
  come from Ultralytics).

### 4. Prediction — rewrite `src/predict.py` + `runner.py:run_predict`
- `model.predict(source)` → top-1 class index → map to `class_values[idx]` (via
  the same `model.names` mapping as eval).

### 5. Config — update `config/default.yaml`
- Add `model: yolo26n-cls.pt`.
- Remove CORN-specific keys (`head`; `label_smoothing` optional — Ultralytics
  supports it via `label_smoothing` train arg, keep if desired).
- Keep: `img_size, epochs, batch_size, learn_rate, weight_decay,
  early_stopping_patience, splits, class_values, num_gpus`.

### 6. Entry point — `main.py`
Keep the three modes (`train` / `evaluate` / `predict`); route them to the new
Ultralytics-based runners. CLI surface unchanged.

## Files removed (full replacement)
- `src/models/backbone.py`, `src/models/heads.py`, `src/models/model.py`
- `src/losses.py`
- `src/training/loop.py`
- `src/data/dataset.py`, `src/data/transforms.py`
- DDP/checkpoint helpers in `src/utils.py` (keep `set_seed` if still useful)

## Files kept
- `src/metrics.py` (verbatim — post-hoc ordinal metrics)
- `src/evaluate.py` (trimmed to `compute_metrics` reporting + confusion matrix)
- `src/logging_setup.py`, `main.py` (rewired)

## Risks / notes
- **Augmentation differs:** Ultralytics' built-in classification augmentation
  replaces the current torchvision v2 transforms; tune via `train()` args.
- **Ordinal signal:** native training uses plain cross-entropy (classes treated
  as nominal); ordinality is recovered only in reporting, not in the loss. This
  is an accepted trade-off of the native approach.
- **DDP:** `num_gpus` maps to the Ultralytics `device` string.
- **License:** Ultralytics YOLO is AGPL-3.0.
- **Checkpoint hygiene:** `yolo26n-cls.pt` auto-downloads to CWD; add `*.pt` and
  `runs/`, `dataset_cls/` to `.gitignore`.
