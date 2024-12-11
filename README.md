# Patch-based Neural Networks with Global Context Integration

Welcome to the GitHub repository for **Patch-based Neural Networks with Global Context Integration**. This repository contains the code and resources for our proposed architecture, which introduces new techniques to enhance patch-based neural networks for improved performance in complex computer vision tasks.

---

## 📄 **Abstract**
Patch-based neural networks have achieved remarkable success due to their computational efficiency and local feature extraction capabilities. However, their reliance on local receptive fields limits the ability to capture global context, essential for understanding complex scenes.

In this paper, we propose an **enhanced architecture** that addresses these limitations through the following contributions:

- **Global Context Network**: Integrates global context to provide a holistic view of the input data.
- **Denoising Head**: Introduced in the bottleneck layer to improve feature robustness and reduce noise.
- **Attention Mechanism**: Added to skip connections to emphasize salient features.

Our experiments on benchmark datasets demonstrate that the proposed modifications lead to **significant performance improvements** over existing architectures.

---

## 🚀 **Key Features**
- **Global Context Network**: Captures both local and global dependencies for better scene understanding.
- **Denoising Head**: Enhances feature robustness by reducing noise within the bottleneck layer.
- **Attention Mechanism**: Uses cross attention mechanism for merging swin transformer local features with global context global features 

---

## 📂 **Repository Structure**
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

## 📊 **Benchmark Results**
<img width="705" alt="benchmark result" src="https://github.com/user-attachments/assets/f1b781f0-c719-4296-bf32-7e32978576c3">

--- 

## 📊 **Visual Comparison**
<img width="931" alt="Visual Comparison 1" src="https://github.com/user-attachments/assets/0fa021d2-d65f-4aee-aae0-46d0bf906fe2">
<img width="931" alt="Visual Comparion 2" src="https://github.com/user-attachments/assets/9b87298b-2a7e-46c1-99d2-48529373109d">

---

