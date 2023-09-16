import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models.detection as detection

from dataset import FlickrLogosDataset


def read_flickr_logos_annotations(annotations_path):
    with open(annotations_path, "r") as f:
        lines = f.readlines()

    # split each line by tab character
    data = [line.strip().split("\t") for line in lines]

    # Extract individual data points from each line
    file_names, class_names, _, x1, y1, x2, y2 = zip(*data)

    return file_names, class_names, x1, y1, x2, y2


file_names, class_names, x1, y1, x2, y2 = read_flickr_logos_annotations("dataset/'flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt")

# Define the dataset and transformations
transform = transforms.Compose([transforms.ToTensor()])
dataset = FlickrLogosDataset(
    "flickr_logos_27_dataset_images",
    file_names,
    class_names,
    list(zip(x1, y1, x2, y2)),
    transforms=transform,
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Initialize the model for Faster R-CNN
num_classes = len(set(class_names)) + 1  # +1 for the background class
model = detection.faster_rcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define the optimizer
optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

print("Training complete!")
