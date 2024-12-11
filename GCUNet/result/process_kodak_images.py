import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def add_gaussian_noise(image, sigma):
    """
    Add Gaussian noise to an image.
    
    Args:
        image (PIL.Image): Input image.
        sigma (int): Standard deviation of the Gaussian noise.
        
    Returns:
        PIL.Image: Noisy image.
    """
    image_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, sigma, image_array.shape).astype(np.float32)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def process_kodak_images(input_dir, output_dir, sigmas):
    """
    Process Kodak images to generate noisy versions.
    
    Args:
        input_dir (str): Path to the folder containing the original Kodak images.
        output_dir (str): Path to the folder where noisy images will be saved.
        sigmas (list): List of noise levels (sigma values).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each noise level
    for sigma in sigmas:
        sigma_output_dir = os.path.join(output_dir, f"sigma{sigma}")
        os.makedirs(sigma_output_dir, exist_ok=True)
        
        for filename in tqdm(os.listdir(input_dir), desc=f"Processing images for sigma={sigma}"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                image = Image.open(input_path).convert('RGB')
                noisy_image = add_gaussian_noise(image, sigma)
                noisy_image.save(os.path.join(sigma_output_dir, filename))

if __name__ == "__main__":
    # Input Kodak24 images directory
    input_directory = "datasets/Kodak24"
    
    # Output directory for noisy images
    output_directory = "datasets/Kodak24_noisy"
    
    # Noise levels
    noise_levels = [10, 30, 50]
    
    # Process the images
    process_kodak_images(input_directory, output_directory, noise_levels)