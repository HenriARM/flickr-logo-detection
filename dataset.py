from torch.utils.data import Dataset
import cv2
import os
import torch


def to_int(s):
    try:
        return int(s)
    except ValueError:
        return None


def str_list_to_int(str_list):
    return [to_int(s) for s in str_list]


def read_flickr_logos_annotations(annotations_path, bbox=True):
    with open(annotations_path, "r") as f:
        lines = f.readlines()

    if bbox:
        # split filename, class name, and bounding box coordinates
        data = [line.strip().split(" ") for line in lines]
        file_names, class_names, _, x1, y1, x2, y2 = zip(*data)
        return (
            file_names,
            class_names,
            str_list_to_int(x1),
            str_list_to_int(y1),
            str_list_to_int(x2),
            str_list_to_int(y2),
        )
    else:
        # split filename, class name
        data = [line.strip().split("\t") for line in lines]
        file_names, class_names = zip(*data)
        return (file_names, class_names)


class FlickrLogosDataset(Dataset):
    # TODO: add strict type hints
    def __init__(self, img_dir, file_names, class_names, box_coords, transforms=None):
        self.img_dir = img_dir
        self.file_names = file_names
        self.class_names = class_names
        self.box_coords = box_coords
        self.transforms = transforms
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(sorted(set(class_names)))
        }

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        x1, y1, x2, y2 = self.box_coords[idx]

        # Check and correct bounding box coordinates
        x1, x2 = min(x1, x2, width - 1), max(x1, x2, width - 1)
        y1, y2 = min(y1, y2, height - 1), max(y1, y2, height - 1)

        # Ensure valid bounding boxes (see example dataset/flickr_logos_27_dataset_images/2662264721.jpg)
        if x2 <= x1:
            x2 = x1 + 1e-5
        if y2 <= y1:
            y2 = y1 + 1e-5

        # Constructing the targets dictionary
        targets = {}
        targets["labels"] = torch.tensor(
            [self.class_to_idx[self.class_names[idx]]], dtype=torch.int64
        )
        targets["boxes"] = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        return image, targets
