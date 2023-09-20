import torch
from torch.optim import SGD
from torch.utils.data import random_split, DataLoader
import torchvision.models as torchmodels
import torchvision.transforms as transforms
import torchvision.models.detection as detection
import torchmetrics
import matplotlib.pyplot as plt

from dataset import FlickrLogosDataset


def to_int(s):
    try:
        return int(s)
    except ValueError:
        return None


def str_list_to_int(str_list):
    return [to_int(s) for s in str_list]


def read_flickr_logos_annotations(annotations_path):
    with open(annotations_path, "r") as f:
        lines = f.readlines()

    # split each line by tab character and space
    data = [line.strip().split("\t") for line in lines]
    data = [line.strip().split(" ") for line in lines]

    # Extract individual data points from each line
    file_names, class_names, _, x1, y1, x2, y2 = zip(*data)

    return (
        file_names,
        class_names,
        str_list_to_int(x1),
        str_list_to_int(y1),
        str_list_to_int(x2),
        str_list_to_int(y2),
    )


# TODO: distractor dataset?
train_annotation_path = "dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
# test_annotation_path = "dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt"
dataset_path = "dataset/flickr_logos_27_dataset_images"

file_names, class_names, x1, y1, x2, y2 = read_flickr_logos_annotations(
    train_annotation_path
)

# Define the dataset and transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((512, 512), antialias=True)]
)
dataset = FlickrLogosDataset(
    dataset_path,
    file_names,
    class_names,
    list(zip(x1, y1, x2, y2)),
    transforms=transform,
)

# Splitting the dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloaders = {}
dataloaders["train"] = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=4
)
dataloaders["val"] = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Initialize the model for Faster R-CNN
num_classes = len(set(class_names)) + 1  # +1 for the background class
model = detection.fasterrcnn_resnet50_fpn(
    weights_backbone=torchmodels.resnet.ResNet50_Weights.IMAGENET1K_V1,
    progress=True,
    num_classes=num_classes,
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
num_epochs = 10
iou_metric = torchmetrics.detection.IntersectionOverUnion()

# List to store epoch-wise losses for plotting
epoch_losses = []
epoch_ious = []

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    running_iou = 0.0

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        for images, targets in dataloaders[phase]:
            images = [image.to(device) for image in images]
            targets = [
                {
                    "labels": targets["labels"][idx].to(device),
                    "boxes": targets["boxes"][idx].to(device),
                }
                for idx in range(len(targets["labels"]))
            ]

            if phase == "train":
                # Calculate training loss
                loss_dict = model(images, targets)
                losses = torch.stack([loss for loss in loss_dict.values()]).sum()
                running_loss += losses.item()

                # Backpropagate
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            else:
                # Calculate IoU for all phases
                with torch.no_grad():
                    outputs = model(images)
                iou = iou_metric(outputs, targets)
                iou = iou["iou"].item()
                if iou:
                    running_iou += iou

                # for image, prediction in zip(images, outputs):
                #    visualize_predictions(image, prediction)

    epoch_loss = running_loss / len(dataloaders["train"])
    epoch_iou = running_iou / len(dataloaders["val"])

    epoch_losses.append(epoch_loss)
    epoch_ious.append(epoch_iou)

    print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}")


# Plotting losses and IoU
plt.figure(figsize=(12, 6))
plt.plot(epoch_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epoch_ious, label="IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("IoU over Epochs")
plt.legend()
plt.grid(True)

# TODO: each epoch plot, augmentation
# TODO: mean average precision, mean average recall, IoU per class