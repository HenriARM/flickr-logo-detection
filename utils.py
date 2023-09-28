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


def visualize_predictions(image, predictions, idx_to_class, score_threshold=0.5):
    """
    Visualize object detection predictions on an image.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        predictions (dict): The model's predictions dictionary.
        score_threshold (float): Minimum confidence score to visualize a prediction.
    """

    image = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    image = np.clip(image, 0, 1)  # Clip to [0, 1] range

    # Name the window and reuse it for plotting
    _, ax = plt.subplots(1, figsize=(10, 8), num="Prediction Visualization")

    # Clear the previous content in the figure window
    ax.imshow(image)

    boxes = predictions.get("boxes")
    labels = predictions.get("labels")
    scores = predictions.get("scores")

    if boxes is None or boxes.numel() == 0:  # Check if boxes tensor is empty
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.clf()  # Clear the current figure
        ax.cla()  # Clear the current axis
        return  # Return as there are no bounding boxes to draw

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            label_text = f"Class: {idx_to_class.get(label.item(), 'Unknown')}, Score: {score.item():.2f}"

            # Plot bounding box
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Add label text
            ax.text(
                x_min,
                y_min,
                label_text,
                bbox=dict(facecolor="red", alpha=0.5),
                fontsize=12,
                color="white",
            )

    plt.axis("off")
    plt.draw()  # Draw the current plot
    plt.waitforbuttonpress(0)  # Wait for a button press
    plt.clf()  # Clear the current figure
    ax.cla()
