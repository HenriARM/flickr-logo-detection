from pathlib import Path
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter


eta_dir = Path("pics/eta")
eta_dir.mkdir(parents=True, exist_ok=True)

annotations_path = "dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
with open(annotations_path, "r") as f:
    lines = f.readlines()

# split each new line
data = [line.strip().split("\t") for line in lines]
# split each new property
data = [line.strip().split(" ") for line in lines]
file_names, class_names, _, x1, y1, x2, y2 = zip(*data)

# plot class distribution
class_counts = Counter(class_names)
plt.figure(figsize=(10, 10))
plt.barh(list(class_counts.keys()), list(class_counts.values()), color="blue")
plt.ylabel("Class Names")
plt.xlabel("Count")
plt.title("Class Distribution")
plt.savefig(eta_dir / "class_distribution.png")
plt.clf()  


# plat picture height and width distribution
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
plt.savefig(eta_dir / "width_height_distribution.png")
plt.clf()  
