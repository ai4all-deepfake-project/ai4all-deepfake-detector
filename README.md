# Deepfake Detection with CNNs – AI4ALL Ignite

Built a deep learning pipeline to detect manipulated video content using the SDFVD 2.0 deepfake dataset. Leveraged OpenCV, PyTorch, and transfer learning (ResNet18 / EfficientNet) to extract facial frames and classify each as real or fake. Developed a full inference pipeline and model interpretability using Grad-CAM, all within AI4ALL’s Ignite accelerator.

## Problem Statement <!--- do not change this line -->

With the increasing realism of AI-generated videos, deepfakes threaten trust in digital media, political stability, and online safety. Accurately detecting such manipulated content is critical for safeguarding public discourse, protecting individuals from misinformation, and enabling content moderation systems.

## Key Results <!--- do not change this line -->

1. Built a complete video classification pipeline that detects deepfakes frame-by-frame and aggregates predictions for video-level inference.
2. Trained a CNN model using ResNet18 and EfficientNet-B0 with over 900 real/fake videos from the SDFVD 2.0 dataset.
3. Achieved over 85% frame-level classification accuracy on validation data.
4. Integrated Grad-CAM to provide visual explanations of the model’s predictions by highlighting manipulated regions in fake frames.

## Methodologies <!--- do not change this line -->

- Extracted and preprocessed frames from videos using OpenCV (resizing, face detection, normalization).
- Labeled frames from SDFVD 2.0 as “real” or “fake” for supervised training.
- Used PyTorch with pretrained CNN architectures (ResNet18 and EfficientNet-B0) for transfer learning.
- Applied `CrossEntropyLoss` and optimized with `AdamW` to reduce overfitting.
- Aggregated frame-level predictions via average fake probability to classify entire videos.
- Applied Grad-CAM to visualize attention maps and interpret model decisions.

## Data Sources <!--- do not change this line -->

- **SDFVD 2.0: Small Scale Deep Fake Video Dataset**  
  A dataset containing 461 real and 461 fake videos with facial augmentations.  
  [Dataset on Mendeley](https://data.mendeley.com/datasets/zzb7jyy8w8/1)

## Technologies Used <!--- do not change this line -->

- Python
- PyTorch
- OpenCV
- TorchVision
- Grad-CAM (`pytorch-gradcam`)
- Google Colab
- Matplotlib / seaborn (for plotting results)

## Authors <!--- do not change this line -->

This project was completed in collaboration with:

- Michael Rosas Ceronio ([GitHub](https://github.com/michaelroscero)
- [Other teammate names and emails or GitHubs here]

Developed as part of the AI4ALL Ignite Program (2025 Cohort).
