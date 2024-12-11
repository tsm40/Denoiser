import os
import argparse
import numpy as np
from skimage.io import imread
from natsort import natsorted
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def evaluate_images(path_gt, path_noise, path_denoise):
    # Get image lists
    gt_images = natsorted([f for f in os.listdir(path_gt) if f.endswith('.png')])
    noise_images = natsorted([f for f in os.listdir(path_noise) if f.endswith('.png')])
    denoise_images = natsorted([f for f in os.listdir(path_denoise) if f.endswith('.png')])

    noise_psnr, denoise_psnr = 0, 0
    noise_ssim, denoise_ssim = 0, 0

    img_num = len(gt_images)

    for j in range(img_num):
        gt_name = gt_images[j]
        noise_name = noise_images[j]
        denoise_name = denoise_images[j]

        gt = imread(os.path.join(path_gt, gt_name)).astype(np.float32) / 255.0
        noise = imread(os.path.join(path_noise, noise_name)).astype(np.float32) / 255.0
        denoise = imread(os.path.join(path_denoise, denoise_name)).astype(np.float32) / 255.0

        gt_g = rgb2gray(gt)
        noise_g = rgb2gray(noise)
        denoise_g = rgb2gray(denoise)

        d_psnr = psnr(gt, denoise, data_range=1.0)
        n_psnr = psnr(gt, noise, data_range=1.0)
        d_ssim = ssim(gt_g, denoise_g, data_range=1.0)
        n_ssim = ssim(gt_g, noise_g, data_range=1.0)

        print(f"GT: {gt_name}")
        print(f"Noise: {noise_name}")
        print(f"Denoise: {denoise_name}")
        print()
        print(f"  Noise PSNR  = {n_psnr:.4f} dB ---------- {j + 1}/{img_num}")
        print(f"  Noise SSIM  = {n_ssim:.4f}     ---------- {j + 1}/{img_num}")
        print(f"Denoise PSNR* = {d_psnr:.4f} dB ---------- {j + 1}/{img_num}")
        print(f"Denoise SSIM* = {d_ssim:.4f}     ---------- {j + 1}/{img_num}")
        print("----------------------------------------------")

        noise_psnr += n_psnr
        denoise_psnr += d_psnr
        noise_ssim += n_ssim
        denoise_ssim += d_ssim

    avg_noise_psnr = noise_psnr / img_num
    avg_denoise_psnr = denoise_psnr / img_num
    avg_noise_ssim = noise_ssim / img_num
    avg_denoise_ssim = denoise_ssim / img_num

    print("                     Finish!                   ")
    print(f"  Average noise PSNR  = {avg_noise_psnr:.4f} dB")
    print(f"  Average noise SSIM  = {avg_noise_ssim:.4f}")
    print(f"Average denoise PSNR* = {avg_denoise_psnr:.4f} dB")
    print(f"Average denoise SSIM* = {avg_denoise_ssim:.4f}")
    print("----------------------------------------------")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate denoising performance.")
    parser.add_argument('--path_gt', type=str, help="Path to ground truth images.",
                        default='./datasets/Ablation/datasets/Kodak24/')
    parser.add_argument('--path_noise', type=str, help="Path to noisy images.",
                        default='./datasets/Ablation/datasets/Kodak24_noisy/sigma50/')
    parser.add_argument('--path_denoise', type=str, help="Path to denoised images.",
                        default='./demo_results/sigma50')
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate_images(args.path_gt, args.path_noise, args.path_denoise)

if __name__ == "__main__":
    main()
