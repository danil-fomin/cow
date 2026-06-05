from torch.utils.data import Dataset
import torchvision.transforms as Transform
from pathlib import Path
from PIL import Image
import torch

class BCSDataset(Dataset):
    def __init__(
        self,
        split: str,
        root: Path = Path("dataset"),
        transforms: Transform.Compose | None = None,
    ):
        self.transforms = transforms
        self.images_dir = root / split / "images"
        self.labels_dir = root / split / "labels"

        self.samples: list[tuple[Path, Path]] = []

        for image_path in sorted(self.images_dir.glob("*.jpg")):
            label_path = self.labels_dir / f"{image_path.stem}.txt"

            if label_path.exists():
                self.samples.append((image_path, label_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        label = int(label_path.read_text().strip())
        label = torch.tensor(label, dtype=torch.long)

        return image, label