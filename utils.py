from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def show_images(images, nrow=8):
    # Make a grid from batch
    img_grid = make_grid(images, nrow=nrow)
    img_np = img_grid.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.axis("off")
    plt.show()


# dataiter = iter(dataloader)
# images, _ = next(dataiter)
# show_images(images, 8)


# plot class distribution

# import matplotlib.pyplot as plt
# from collections import Counter
# class_counts = Counter(class_names)
# plt.figure(figsize=(10,10))
# plt.barh(list(class_counts.keys()), list(class_counts.values()), color='blue')
# plt.ylabel('Class Names')
# plt.xlabel('Count')
# plt.title('Class Distribution')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def visualize_predictions(image, predictions, score_threshold=0.5):
    """
    Visualize object detection predictions on an image.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        predictions (dict): The model's predictions dictionary.
        score_threshold (float): Minimum confidence score to visualize a prediction.

    Returns:
        None (displays the image with bounding boxes and labels).
    """
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    image = np.clip(image, 0, 1)  # Clip to [0, 1] range

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            label_text = f"Class: {label.item()}, Score: {score.item():.2f}"

            # Plot bounding box
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add label text
            ax.text(
                x_min,
                y_min,
                label_text,
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=12,
                color='white'
            )

    plt.axis('off')
    plt.show()