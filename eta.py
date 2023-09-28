annotations_path = "dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
from pathlib import Path

eta_dir = Path("pics/eta")
eta_dir.mkdir(parents=True, exist_ok=True)

with open(annotations_path, "r") as f:
    lines = f.readlines()

# split each new line
data = [line.strip().split("\t") for line in lines]
# split each new property
data = [line.strip().split(" ") for line in lines]
file_names, class_names, _, x1, y1, x2, y2 = zip(*data)

# plot class distribution
import matplotlib.pyplot as plt
from collections import Counter

class_counts = Counter(class_names)
plt.figure(figsize=(10, 10))
plt.barh(list(class_counts.keys()), list(class_counts.values()), color="blue")
plt.ylabel("Class Names")
plt.xlabel("Count")
plt.title("Class Distribution")
plt.savefig(eta_dir / "class_distribution.png")
