
# Download dataset

```.bash
$ chmod +x download_and_extract.sh
$ ./download_and_extract.sh
```

# Download checkpoint
```.bash
$ inference.py
```

TODO: reqs txt


# Results

![IoU](pics/experiment1/class_iou_epochs.png)
![Loss](pics/experiment1/loss_epochs.png)
![mAP](pics/experiment1/map_epochs.png)

# EDA

Class distribution

![Class distribution](pics/eta/class_distribution.png)

Image resolution distribution

![Width height distribution](pics/eta/image_resolution_distribution.png)

Bounding box size distribution

![Width height distribution](pics/eta/bounding_box_size_distribution.png)
