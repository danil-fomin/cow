import random
from pathlib import Path
from collections import defaultdict
import shutil
import yaml

def extract_samples(input_dir: Path) -> list[tuple[Path, str]]:
    samples = []

    for directories in input_dir.iterdir():
        for directory in directories.iterdir():
            label = directory.name

            for image_path in directory.iterdir():
                samples.append((image_path, label))

    return samples


def build_dataset(
    samples: list[tuple[Path, str]],
    splits: dict[str, float],
    class_values: list,
    output_dir: Path = Path("dataset"),
    seed: int = 42,
):
    total = sum(splits.values())
    ratios = {name: coef / total for name, coef in splits.items()}

    label_to_index = {str(value): i for i, value in enumerate(class_values)}

    files_by_label: dict[str, list[Path]] = defaultdict(list)
    for file, label in samples:
        files_by_label[label].append(file)

    rng = random.Random(seed)
    counters = {name: 0 for name in ratios}
    split_names = list(ratios.keys())

    for label, files in files_by_label.items():
        if label not in label_to_index:
            continue

        class_idx = label_to_index[label]

        files = sorted(files, key=lambda f: f.name)
        rng.shuffle(files)
        n = len(files)

        start = 0
        for i, name in enumerate(split_names):
            if i == len(split_names) - 1:
                chunk = files[start:]
            else:
                count = int(n * ratios[name])
                chunk = files[start : start + count]
                start += count

            for file in chunk:
                _place_file(file, class_idx, name, counters[name], output_dir)
                counters[name] += 1


def _place_file(file: Path, class_idx: int, split: str, index: int, output_dir: Path):
    images_dir = output_dir / split / "images"
    labels_dir = output_dir / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"{index}.png"
    label_path = labels_dir / f"{index}.txt"

    shutil.copy(file, image_path)
    with open(label_path, "w+") as f:
        f.write(f"{class_idx}\n")


if __name__ == "__main__":
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    splits = config.get("splits")
    samples = extract_samples(Path("dataset/knee-osteoarthritis-dataset-with-severity"))
    build_dataset(samples, splits, config["class_values"])
