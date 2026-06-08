import random
import shutil
from pathlib import Path

import yaml

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def _split_files(files: list[Path], ratios: dict[str, float], seed: int) -> dict[str, list[Path]]:
    files = sorted(files, key=lambda f: f.name)
    random.Random(seed).shuffle(files)

    n = len(files)
    split_names = list(ratios)
    result: dict[str, list[Path]] = {}
    start = 0
    for i, name in enumerate(split_names):
        if i == len(split_names) - 1:
            result[name] = files[start:]
        else:
            count = int(n * ratios[name])
            result[name] = files[start : start + count]
            start += count
    return result


def split_dataset(
    source_dir: Path,
    splits: dict[str, float],
    output_dir: Path = Path("dataset_cls"),
    seed: int = 42,
) -> dict[str, int]:
    """Reshape ``source_dir/<class>/*.img`` into the Ultralytics classification
    layout ``output_dir/{split}/<class>/*.img``, preserving class folder names."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    total = sum(splits.values())
    ratios = {name: coef / total for name, coef in splits.items()}

    counts = {name: 0 for name in ratios}
    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    for class_dir in class_dirs:
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        for split_name, chunk in _split_files(files, ratios, seed).items():
            dest_dir = output_dir / split_name / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file in chunk:
                shutil.copy(file, dest_dir / file.name)
                counts[split_name] += 1
    return counts


if __name__ == "__main__":
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)
    result = split_dataset(Path("dataset/dataset"), config["splits"], Path(config["data_dir"]))
    print(result)
