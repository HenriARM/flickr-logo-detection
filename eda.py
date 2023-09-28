from pathlib import Path
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from dataset import FlickrLogosDataset, read_flickr_logos_annotations


eta_dir = Path("pics/eta")
eta_dir.mkdir(parents=True, exist_ok=True)

annotations_path = "dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
with open(annotations_path, "r") as f:
    lines = f.readlines()

file_names, class_names, x1, y1, x2, y2 = read_flickr_logos_annotations(
    annotations_path
)

# 1. plot class distribution
class_counts = Counter(class_names)
plt.figure(figsize=(10, 10))
plt.barh(list(class_counts.keys()), list(class_counts.values()), color="blue")
plt.ylabel("Class Names")
plt.xlabel("Count")
plt.title("Class Distribution")
plt.savefig(eta_dir / "class_distribution.png")


# 2. plot picture height and width distribution
dataset_path = Path("dataset/flickr_logos_27_dataset_images")
heights = []
widths = []
for img_path in dataset_path.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    heights.append(h)
    widths.append(w)

_, axs = plt.subplots(1, 2, figsize=(12, 6))

# plot histogram for heights
axs[0].hist(heights, bins=30, color="blue", edgecolor="black")
axs[0].set_title("Histogram of Heights")
axs[0].set_xlabel("Height")
axs[0].set_ylabel("Number of Images")

# plot histogram for widths
axs[1].hist(widths, bins=30, color="red", edgecolor="black")
axs[1].set_title("Histogram of Widths")
axs[1].set_xlabel("Width")
axs[1].set_ylabel("Number of Images")

plt.tight_layout()
plt.savefig(eta_dir / "image_resolution_distribution.png")
plt.clf()


# 3. plot box average width and height per class

dataset = FlickrLogosDataset(
    dataset_path,
    file_names,
    class_names,
    list(zip(x1, y1, x2, y2)),
    transforms=None,
)

class_shapes = {}
for class_name in set(dataset.class_names):
    class_shapes[class_name] = {"width": 0, "height": 0, "count": 0}

# iterate over the dataset
for i in range(len(dataset)):
    _, target = dataset[i]
    class_name = dataset.class_names[i]
    box = target["boxes"][0].numpy()  # assuming one box per image
    width = box[2] - box[0]
    height = box[3] - box[1]

    class_shapes[class_name]["width"] += width
    class_shapes[class_name]["height"] += height
    class_shapes[class_name]["count"] += 1

# calculate the average width and height for each class
avg_shapes = {}
for class_name, values in class_shapes.items():
    avg_width = values["width"] / values["count"]
    avg_height = values["height"] / values["count"]
    avg_shapes[class_name] = {"avg_width": avg_width, "avg_height": avg_height}

    print(
        f"Class: {class_name}, Average Width: {avg_width}, Average Height: {avg_height}"
    )

class_names = list(avg_shapes.keys())
avg_widths = [avg_shapes[class_name]["avg_width"] for class_name in class_names]
avg_heights = [avg_shapes[class_name]["avg_height"] for class_name in class_names]

plt.figure(figsize=(10, 6))
plt.bar(class_names, avg_widths, alpha=0.5, label="Avg Width")
plt.bar(class_names, avg_heights, alpha=0.5, label="Avg Height")
plt.xlabel("Class Name")
plt.ylabel("Pixel")
plt.title("Average Width and Height per Class")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(eta_dir / "bounding_box_size_distribution.png")


# TODO: add more to EDA
# Aspect Ratios of Bounding Boxes
# Number of Objects per Image
# Spatial Distribution of Logos
# Color Distribution of Logos
# Sample Visualization
# Data Quality: Check for any inconsistencies in the dataset, such as missing labels, incorrect bounding boxes, etc.