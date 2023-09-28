from tqdm import tqdm
from pathlib import Path
import torch
import json
from torch.optim import SGD

# TODO: visualise lr
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader
import torchvision.models as torchmodels
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
import matplotlib

matplotlib.use("Agg")  # Set a non-GUI backend
import matplotlib.pyplot as plt

plt.ioff()

from dataset import FlickrLogosDataset, read_flickr_logos_annotations


def combine_predictions(standard_preds, tta_preds):
    combined_preds = []
    for std_pred, tta_pred in zip(standard_preds, tta_preds):
        combined_pred = {
            "boxes": std_pred["boxes"],
            "scores": (std_pred["scores"] + tta_pred["scores"]) / 2,
        }
        combined_preds.append(combined_pred)
    return combined_preds


train_annotation_path = "dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
dataset_path = "dataset/flickr_logos_27_dataset_images"

file_names, class_names, x1, y1, x2, y2 = read_flickr_logos_annotations(
    train_annotation_path
)

# save class names to use in inference
with open("class_to_idx.json", "w") as f:
    class_to_idx = {
        class_name: idx for idx, class_name in enumerate(sorted(set(class_names)))
    }
    json.dump(class_to_idx, f)

dataset = FlickrLogosDataset(
    dataset_path,
    file_names,
    class_names,
    list(zip(x1, y1, x2, y2)),
    transforms=None,
)


# Splitting the dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

image_shape = (512, 512)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(image_shape, antialias=True),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(image_shape, antialias=True),
        # TODO: keep_aspect_ratio
        # transforms.Pad(padding_mode="edge"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomCrop((400, 400)),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.RandomGrayscale(p=0.1),
    ]
)

tta_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=1),
    ]
)

# Apply the transformations to the datasets
train_dataset.dataset.transforms = train_transform
val_dataset.dataset.transforms = val_transform

batch_size = 8
dataloaders = {
    "train": DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    ),
    "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
}

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
num_epochs = 100

iou_metric = IntersectionOverUnion()
map_metric = MeanAveragePrecision()

# List to store epoch-wise losses for plotting
epoch_losses = []
epoch_ious = []
epoch_class_ious = {i: [] for i in range(num_classes)}
epoch_maps = []

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    running_iou = 0.0
    running_class_ious = {i: [] for i in range(num_classes)}
    running_map = 0.0

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()
            # TODO: iou_metric.reset()
            map_metric.reset()

            tta_preds = []
            tta_targets = []

        for images, targets in tqdm(dataloaders[phase], desc=phase, ncols=100):
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

                    # tta_images = [tta_transform(image) for image in images]
                    # tta_outputs = model(tta_images)

                    # tta_preds.extend(tta_outputs)
                    # tta_targets.extend(targets)

                iou = iou_metric(outputs, targets)
                iou = iou["iou"].item()
                if iou:
                    running_iou += iou

                map_metric.update(outputs, targets)

                for pred, target in zip(outputs, targets):
                    for gt_box, gt_label in zip(target["boxes"], target["labels"]):
                        for pred_box, pred_label in zip(pred["boxes"], pred["labels"]):
                            if gt_label == pred_label:
                                iou = iou_metric([pred], [target])
                                iou = iou["iou"].item()
                                if iou:
                                    running_class_ious[gt_label.item()].append(iou)

        # if phase == "val":
        #     combined_preds = combine_predictions(outputs, tta_preds)

    scheduler.step()

    epoch_loss = running_loss / len(dataloaders["train"])
    epoch_iou = running_iou / len(dataloaders["val"])
    epoch_map = map_metric.compute()["map"].item()

    epoch_losses.append(epoch_loss)
    epoch_ious.append(epoch_iou)
    epoch_maps.append(epoch_map)

    mean_ious_per_class = {
        cls: sum(ious) / len(ious) if ious else 0
        for cls, ious in running_class_ious.items()
    }

    for cls, mean_iou in mean_ious_per_class.items():
        epoch_class_ious[cls].append(mean_iou)

    print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}")

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = checkpoints_dir / f"model_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "iou": epoch_iou,
            },
            checkpoint_path,
        )

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plotting losses and save
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss over Epochs up to Epoch {epoch + 1}")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "loss_epochs.png")
    plt.close()  # Close the plot to free up memory

    # Plotting IoU and save
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_ious, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title(f"IoU over Epochs up to Epoch {epoch + 1}")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "iou_epochs.png")
    plt.close()  # Close the plot to free up memory

    # Plotting Classwise IoU and save
    plt.figure(figsize=(12, 6))
    for cls, ious in epoch_class_ious.items():
        plt.plot(ious, label=f"Class {cls}")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Class-wise IoU over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "class_iou_epochs.png")
    plt.close()  # Close the plot to free up memory

    # Plotting mAP and save
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_maps, label="mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Mean Average Precision over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "map_epochs.png")
    plt.close()  # Close the plot to free up memory

# TODO: mean average recall
# TODO: log training
