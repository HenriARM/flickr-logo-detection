#!/bin/bash

# Create dataset directory if it doesn't exist
mkdir -p dataset

# Download the dataset tar.gz file
wget http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz

# Extract the main dataset file into the dataset/ directory
tar -xzvf flickr_logos_27_dataset.tar.gz -C dataset/

# Extract the images tar.gz file into the dataset/ directory
tar -xzvf dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz -C dataset/
