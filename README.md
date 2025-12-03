# Tiny ImageNet & CIFAR-10 Classification with PyTorch & Ignite

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![License](https://img.shields.io/badge/License-MIT-green)

A complete PyTorch pipeline for image classification on **Tiny ImageNet** and **CIFAR-10** datasets using **custom CNNs** and **pre-trained models**. The project includes data preprocessing, augmentation, GPU support, and evaluation with **PyTorch Ignite**.

---

## Features
- Dataset preprocessing, augmentation, and DataLoader setup  
- Validation folder organization for Tiny ImageNet  
- Custom Convolutional Neural Network implementation  
- Training and evaluation with PyTorch Ignite  
- Metrics: **Accuracy, Loss, Precision, Recall, F1-score**  
- GPU acceleration support  
- Experiments with **pre-trained ImageNet normalization** and **EfficientNet**

## Visualization

**Batch of Images:** ![Batch of Image](https://github.com/Muskaan322/Tiny_ImageNet_and_CIFAR-10_Classification-_with-_PyTorch_and_Ignite/blob/main/batch%20of%20training%20set%20image.png)

**Batch of Pre-Trained Normalized Image:** ![Batch of pre-trained normalized image](https://github.com/Muskaan322/Tiny_ImageNet_and_CIFAR-10_Classification-_with-_PyTorch_and_Ignite/blob/main/batch%20of%20pre-train%20normalized%20image.png)

**Batch of Validation:** ![Batch of Validation](https://github.com/Muskaan322/Tiny_ImageNet_and_CIFAR-10_Classification-_with-_PyTorch_and_Ignite/blob/main/Display%20batch%20of%20validation.png)

  ## Results

### CIFAR-10 (Custom CNN)
| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.72  |
| Loss      | 1.05  |
| Precision | 0.71  |
| Recall    | 0.72  |
| F1-score  | 0.71  |

### Tiny ImageNet (Pre-trained EfficientNet)
| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.54  |
| Loss      | 2.10  |
| Precision | 0.53  |
| Recall    | 0.54  |
| F1-score  | 0.53  |

> ⚠️ Note: Metrics may vary depending on GPU, batch size, and number of epochs. 

---

## Installation & Setup
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install torch torchvision torchaudio
pip install pytorch-ignite efficientnet_pytorch opendatasets matplotlib pandas numpy

