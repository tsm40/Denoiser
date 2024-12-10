# Evaluate Pre-trained Models
Follow these steps to evaluate the performance of the best pre-trained models:
1. The `result` folder is used to evaluate the best pre-trained models.
2. Run `download_datasets.sh` to download the tetsing datasets.
3. Execute `process_kodak_images.py` to add Gaussian noise to the images.
4. Once the images are downloaded and processed, they are ready for testing. Open the `GCNet Evaluation.ipynb` in the home directory to generate denoised images and compute evaluation metrics.
