import os
import requests
from tqdm import tqdm

# URL and directories
base_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
datasets = {
    "train_HR": "DIV2K_train_HR.zip",  # High-Resolution images
    "train_LR_bicubic_X2": "DIV2K_train_LR_bicubic_X2.zip",  # Low-Resolution images (Bicubic X2)
    "train_LR_bicubic_X3": "DIV2K_train_LR_bicubic_X3.zip",  # Low-Resolution images (Bicubic X3)
    "train_LR_bicubic_X4": "DIV2K_train_LR_bicubic_X4.zip",  # Low-Resolution images (Bicubic X4)
    "valid_HR": "DIV2K_valid_HR.zip",  # Validation High-Resolution images
    "valid_LR_bicubic_X2": "DIV2K_valid_LR_bicubic_X2.zip",  # Validation Low-Resolution images (Bicubic X2)
    "valid_LR_bicubic_X3": "DIV2K_valid_LR_bicubic_X3.zip",  # Validation Low-Resolution images (Bicubic X3)
    "valid_LR_bicubic_X4": "DIV2K_valid_LR_bicubic_X4.zip",  # Validation Low-Resolution images (Bicubic X4)
}

# Folder to save the dataset
output_folder = "DIV2K"
os.makedirs(output_folder, exist_ok=True)

# Function to download a file with a progress bar
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as file, tqdm(
        desc=output_path,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(1024):
            bar.update(len(data))
            file.write(data)

# Download each dataset
for name, filename in datasets.items():
    url = base_url + filename
    output_path = os.path.join(output_folder, filename)
    print(f"Downloading {name}...")
    download_file(url, output_path)
    print(f"{name} downloaded successfully.")

print("All downloads completed.")
