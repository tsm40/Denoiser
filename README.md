# Patch-based Neural Networks with Global Context Integration
Welcome to the GitHub repository for **Patch-based Neural Networks with Global Context Integration**. This repository contains the code and resources for our proposed architecture, which introduces new techniques to enhance patch-based neural networks for improved performance in complex computer vision tasks.


## 📄 **Abstract**
Patch-based neural networks have achieved remarkable success due to their computational efficiency and local feature extraction capabilities. However, their reliance on local receptive fields limits the ability to capture global context, essential for understanding complex scenes.

In this paper, we propose an **enhanced architecture** that addresses these limitations through the following contributions:

- **Global Context Network**: Integrates global context to provide a holistic view of the input data.
- **Denoising Head**: Introduced in the bottleneck layer to improve feature robustness and reduce noise.
- **Attention Mechanism**: Added to skip connections to emphasize salient features.

Our experiments on benchmark datasets demonstrate that the proposed modifications lead to **significant performance improvements** over existing architectures.


## 🚀 **Key Features**
- **Global Context Network**: Captures both local and global dependencies for better scene understanding.
- **Denoising Head**: Enhances feature robustness by reducing noise within the bottleneck layer.
- **Attention Mechanism**: Uses cross attention mechanism for merging swin transformer local features with global context global features 


## 📂 **Repository Structure**
The Global-Local Optimization Network (GLOWNet) combines the Global Context Network (GCN) with a patch-based architecture to bridge local and global representations. The Global Context UNet (GCUNet), using GCN within a UNet architecture, serves as our baseline.
```plaintext
patch-based-global-context-network/
├── GLOWNet
    ├── datasets/          # Scripts to download and preprocess datasets
    ├── models/            # Model architecture implementations
    ├── utils/             # Helper functions and utilities
    ├── warmup_scheduler/  # Warmup of learning rate during training
    ├── other files        # Other training and experiment files    
├── GCUNet
    ├── datasets/          # Scripts to download and preprocess datasets
    ├── models/            # Model architecture implementations
    ├── utils/             # Helper functions and utilities
    ├── warmup_scheduler/  # Warmup of learning rate during training
    ├── other files        # Other training and experiment files
├── README.md              # Project overview
```


## 🐇 **Quick Run**
To test our [pre-trained model](https://drive.google.com/file/d/19YrFIHw0todZ5O7c1H0XUOIFkF6V8WU1/view?usp=drive_link) (S-GLOWNet, which was the best performing in our experiment) on noisy images, run
```
python demo_any_resolution.py --input_dir noisy_images_folder_path --stride shifted_window_stride --result_dir denoised_images_folder_path --weights path_to_models
```


## 📊 **Benchmark Results**
<img width="710" alt="Benchmark Result" src="https://github.com/user-attachments/assets/47387d91-9bc3-4c4d-8bd8-bc87d6589ba4" />


## 🏞️ **Visual Comparison**
<img width="935" alt="Visual Comparison 1" src="https://github.com/user-attachments/assets/9239a39e-7673-4b57-9b66-6b92eb56a76c" />
<img width="935" alt="Visual Comparison 2" src="https://github.com/user-attachments/assets/0263f834-94f1-4143-a59a-43a6bf4849d3" />

