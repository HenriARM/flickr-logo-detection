from torch.utils.data import Dataset
from PIL import Image
import os
import torch


class FlickrLogosDataset(Dataset):
    # TODO: add strict type hints
    def __init__(self, img_dir, file_names, class_names, box_coords, transforms=None):
        self.img_dir = img_dir
        self.file_names = file_names
        self.class_names = class_names
        self.box_coords = box_coords
        self.transforms = transforms
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(set(class_names))
        }

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        image = Image.open(img_path).convert("RGB")

        # Constructing the targets dictionary
        targets = {}
        targets["labels"] = torch.tensor(
            [self.class_to_idx[self.class_names[idx]]], dtype=torch.int64
        )
        targets["boxes"] = torch.tensor([self.box_coords[idx]], dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        return image, targets
