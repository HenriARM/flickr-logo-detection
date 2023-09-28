import cv2
import json
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from pathlib import Path
from utils import visualize_predictions
from dataset import read_flickr_logos_annotations

with open("class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

# Create inverse mapping
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
num_classes = len(class_to_idx) + 1

annotations_path = (
    "dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt"
)
file_names, class_names = read_flickr_logos_annotations(annotations_path, bbox=False)

# TODO: put checkpoint_path, image_path, and num_classes as args
dataset_path = Path("dataset/flickr_logos_27_dataset_images")

# Load the checkpoint
checkpoint_path = Path("checkpoints/model_epoch_100.pth")
checkpoint = torch.load(checkpoint_path)

# Load the model and set it to evaluation mode
model = detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Move the model to the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

batch_size = 8
image_paths = list(dataset_path.glob("*.jpg"))
for i in range(0, len(list(dataset_path.glob("*.jpg"))), batch_size):
    batch_image_paths = image_paths[i : i + batch_size]
    batch_images = []

    # Load and preprocess images in the batch
    for image_path in batch_image_paths:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        batch_images.append(image)

    # Move images to device
    batch_images = [image.to(device) for image in batch_images]

    # Run inference on the batch
    with torch.no_grad():
        predictions = model(batch_images)

    # Process or visualize the predictions as needed
    for j, prediction in enumerate(predictions):
        print(f"\nPrediction for {batch_image_paths[j]}: ", prediction)
        visualize_predictions(batch_images[j], prediction, idx_to_class)
