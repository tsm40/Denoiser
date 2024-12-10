#!/bin/bash

# Create directories for datasets
mkdir -p datasets/Kodak24
mkdir -p datasets/CBSD68

# Download the Kodak dataset directly into the datasets directory
echo "Downloading Kodak24 dataset from Kaggle..."
curl -L -o datasets/Kodak24/kodak24.zip \
https://www.kaggle.com/api/v1/datasets/download/sherylmehta/kodak-dataset

# Extract the dataset
echo "Extracting Kodak24 dataset..."
unzip -q datasets/Kodak24/kodak24.zip -d datasets/Kodak24
rm datasets/Kodak24/kodak24.zip

echo "Kodak24 dataset downloaded and extracted successfully."

# Download CBSD68 dataset
echo "Downloading CBSD68 dataset..."
wget -q -O datasets/CBSD68/CBSD68.tar.gz https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.tar.gz
tar -xzf datasets/CBSD68/CBSD68.tar.gz -C datasets/CBSD68 --strip-components=1
rm datasets/CBSD68/CBSD68.tar.gz
echo "CBSD68 dataset downloaded and extracted."