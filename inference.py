import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from pathlib import Path

from utils import visualize_predictions

# TODO: put checkpoint_path, image_path, and num_classes as args

# Load the checkpoint
checkpoint_path = Path("checkpoints/model_epoch_100.pth")
checkpoint = torch.load(checkpoint_path)

# Load the model and set it to evaluation mode
num_classes = 28
model = detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Move the model to the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Load an arbitrary image with OpenCV
image_path = "dataset/flickr_logos_27_dataset_images/123937306.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert BGR image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform the necessary transformations on the image
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
image = transform(image)
image = image.to(device)

# Perform inference
with torch.no_grad():
    prediction = model([image])

print(prediction)
visualize_predictions(image, prediction)