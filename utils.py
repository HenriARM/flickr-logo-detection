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
