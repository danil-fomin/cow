from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def gather_images(path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    return [path]


def predict_paths(model, paths: list[Path], index_to_value: dict):
    """Return list of (path, class_index, bcs_value) for each image."""
    results = []
    predictions = model.predict([str(p) for p in paths], verbose=False)
    for path, result in zip(paths, predictions):
        index = int(result.probs.top1)
        results.append((path, index, index_to_value[index]))
    return results
