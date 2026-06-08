import random
import shutil
from pathlib import Path
from xml.etree import ElementTree

import yaml

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def parse_voc(xml_path: Path):
    root = ElementTree.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip()
        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        objects.append((name, (xmin, ymin, xmax, ymax)))

    return width, height, objects


def _name_to_index(name: str, sorted_values: list) -> int | None:
    try:
        value = float(name)
    except ValueError:
        return None
    for index, sorted_value in enumerate(sorted_values):
        if abs(value - sorted_value) < 1e-6:
            return index
    return None


def voc_to_yolo_lines(objects, width: int, height: int, sorted_values: list) -> list[str]:
    lines = []
    for name, (xmin, ymin, xmax, ymax) in objects:
        index = _name_to_index(name, sorted_values)
        if index is None:
            continue
        cx = ((xmin + xmax) / 2) / width
        cy = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _find_image(xml_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = xml_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def _assign_splits(pairs: list, splits: dict, seed: int) -> dict:
    total = sum(splits.values())
    ratios = {name: coef / total for name, coef in splits.items()}
    split_names = list(ratios)

    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)

    assignments = {}
    start = 0
    for i, name in enumerate(split_names):
        if i == len(split_names) - 1:
            chunk = pairs[start:]
        else:
            count = int(n * ratios[name])
            chunk = pairs[start : start + count]
            start += count
        for pair in chunk:
            assignments[pair] = name
    return assignments


def build_detection_dataset(
    source_dir: Path,
    splits: dict,
    class_values: list,
    output_dir: Path,
    seed: int = 42,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_values = sorted(class_values)

    pairs = []
    for xml_path in sorted(source_dir.rglob("*.xml")):
        image_path = _find_image(xml_path)
        if image_path is not None:
            pairs.append((image_path, xml_path))

    assignments = _assign_splits(pairs, splits, seed)

    counts = {name: 0 for name in splits}
    for (image_path, xml_path), split_name in assignments.items():
        width, height, objects = parse_voc(xml_path)
        lines = voc_to_yolo_lines(objects, width, height, sorted_values)

        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(image_path, image_dir / image_path.name)
        label_text = "\n".join(lines)
        (label_dir / f"{image_path.stem}.txt").write_text(label_text + "\n" if lines else "")
        counts[split_name] += 1

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: str(value) for index, value in enumerate(sorted_values)},
    }
    data_path = output_dir / "data.yaml"
    with open(data_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return data_path, counts


if __name__ == "__main__":
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    data_path, counts = build_detection_dataset(
        source_dir=config["source"],
        splits=config["splits"],
        class_values=config["class_values"],
        output_dir=config["data_dir"],
        seed=config.get("seed", 42),
    )
    print(f"data.yaml: {data_path}")
    print(counts)
