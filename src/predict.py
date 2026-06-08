from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def gather_images(path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    return [path]
