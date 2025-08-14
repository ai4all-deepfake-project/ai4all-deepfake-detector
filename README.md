# Deepfake Detection with CNNs – AI4ALL Ignite

A deep learning–based system for classifying real and deepfake videos using the EfficientNet-B0 backbone.  
The model processes video frame-by-frame, aggregates predictions, and outputs a probability score for deepfakes.  

## Features
- **Face-focused preprocessing** to target identity-specific features.
- **EfficientNet-B0** CNN architecture for efficient, high-accuracy classification.
- **Video inference pipeline** for frame extraction, prediction, and aggregation.
- Supports **MP4/H.264 videos** for easy compatibility.

## Dataset
- Based on **Small-scale Deepfake Forgery Video Dataset (SDFVD) 2.0**.
- Contains real and deepfake videos with built-in augmentation (rotation, scaling, brightness/contrast adjustments).
- Preprocessing crops frames to facial regions to improve detection.

## Installation
```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
````

## Usage

### Training

```bash
python train.py
```

### Inference

Run the Gradio app for video upload and prediction:

```bash
python app.py
```

Or in Colab:

```python
!pip install -r requirements.txt
!python app.py
```

## Example

Upload a video and get:

```
Video is likely Fake (Average Fake Probability: 0.7321)
```

## Next Steps

* Improve accuracy with cleaner, more balanced datasets.
* Experiment with additional fine-tuning of backbone layers.
* Expand dataset diversity for better generalization.

