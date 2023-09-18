import torch
from torch.optim import SGD
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import torchvision.models.detection as detection
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
annotation_path = "dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
dataset_path = "dataset/flickr_logos_27_dataset_images"
file_names, class_names, x1, y1, x2, y2 = read_flickr_logos_annotations(annotation_path)

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
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Initialize the model for Faster R-CNN
num_classes = len(set(class_names)) + 1  # +1 for the background class
model = detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define the optimizer
optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Number of training epochs
num_epochs = 10

# List to store epoch-wise losses for plotting
train_epoch_losses = []
val_epoch_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_total_loss = 0.0

    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        targets = [
            {
                "labels": targets["labels"][idx].to(device),
                "boxes": targets["boxes"][idx].to(device),
            }
            for idx in range(len(targets["labels"]))
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_total_loss += losses.item()

    train_avg_loss = train_total_loss / len(train_dataloader)
    train_epoch_losses.append(train_avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_avg_loss}")

    # Validation
    model.eval()
    val_total_loss = 0.0
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = list(image.to(device) for image in images)
            targets = [
                {
                    "labels": targets["labels"][idx].to(device),
                    "boxes": targets["boxes"][idx].to(device),
                }
                for idx in range(len(targets["labels"]))
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_total_loss += losses.item()

    val_avg_loss = val_total_loss / len(val_dataloader)
    val_epoch_losses.append(val_avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_avg_loss}")

print("Training complete!")

# Plotting the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_epoch_losses, label="Training Loss", color="blue")
plt.plot(val_epoch_losses, label="Validation Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
