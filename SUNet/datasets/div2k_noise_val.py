import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def add_gaussian_noise(image, var):
    """
    Adds Gaussian noise to an image.

    Parameters:
        image (numpy.ndarray): Input image in uint8 format.
        var (float): Variance of the Gaussian noise.

    Returns:
        noisy_image (numpy.ndarray): Noisy image in uint8 format.
    """
    # Normalize the image to [0, 1]
    image_normalized = image.astype(np.float32) / 255.0

    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(var), image_normalized.shape).astype(np.float32)

    # Add noise to the image
    noisy_image = image_normalized + noise

    # Clip the values to [0, 1] and convert back to [0, 255]
    noisy_image = np.clip(noisy_image, 0, 1) * 255.0
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image

def ensure_directory(path):
    """
    Ensures that a directory exists. If it does not exist, creates it.

    Parameters:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # Define input and output directories using Linux-style paths
    file_path = '/home/shared/Denoiser/SUNet/datasets/DIV2K/DIV2K_valid_HR/'        # Clean images directory
    target_path = '/home/shared/Denoiser/SUNet/datasets/Denoising_DIV2K/test/target/' # Clean patches output
    input_path = '/home/shared/Denoiser/SUNet/datasets/Denoising_DIV2K/test/input/'      # Noisy patches output

    # Ensure output directories exist
    ensure_directory(target_path)
    ensure_directory(input_path)

    # Get list of all PNG files in the clean images directory
    img_path_list = glob(os.path.join(file_path, '*.png'))
    img_num = len(img_path_list)
    count = 0

    if img_num > 0:
        print(f"Found {img_num} images in {file_path}. Processing...")

        # Iterate over each image with a progress bar
        for j, img_path in enumerate(tqdm(img_path_list, desc="Processing Images"), 1):
            # Read the image
            input_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if input_image is None:
                print(f"Warning: Unable to read {img_path}. Skipping.")
                continue

            # Image dimensions
            X, Y, _ = input_image.shape

            # Parameters
            a = 255  # Width of the crop
            b = 255  # Height of the crop
            c = 3    # Number of crops per image

            # Check if the image is large enough
            if X <= a + 1 or Y <= b + 1:
                print(f"Warning: Image {img_path} is smaller than the crop size. Skipping.")
                continue

            for _ in range(c):
                # Randomly select top-left corner for cropping
                y = np.random.randint(0, X - a)
                x = np.random.randint(0, Y - b)

                # Crop the image
                C = input_image[y:y+a, x:x+b]

                # Define noise levels
                sigmas = [10, 30, 50]
                variances = [(sig / 256.0) ** 2 for sig in sigmas]

                for sig, var in zip(sigmas, variances):
                    # Add Gaussian noise
                    added_noise = add_gaussian_noise(C, var)

                    # Increment count
                    count += 1

                    # Define filenames
                    noisy_filename = os.path.join(input_path, f"{count}.png")
                    clean_filename = os.path.join(target_path, f"{count}.png")

                    # Save the noisy and clean images
                    cv2.imwrite(noisy_filename, added_noise)
                    cv2.imwrite(clean_filename, C)

            print(f"Processed Image {j}/{img_num}: {os.path.basename(img_path)}")

    print("Finished processing all images!")

if __name__ == "__main__":
    main()
