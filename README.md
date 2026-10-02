# Deepfake Detection with a CNN-GRU

This project classifies a video as **REAL** or **FAKE**. A pretrained ResNet50 extracts visual features from up to 20 sampled frames, and a pair of GRU layers learns temporal patterns from those frame features. During training, horizontal flips, small rotations, noise, brightness, and contrast augmentation are applied to real training videos only.

## Requirements

- Python 3.10 or 3.11
- TensorFlow 2.15
- The Kaggle Deepfake Detection Challenge sample videos
- Internet access the first time ResNet50 ImageNet weights are requested

Python dependencies are listed in `requirements.txt`.

## Dataset

The training command expects this layout:

```text
data/deepfake-detection-challenge/
├── train_sample_videos/
│   ├── metadata.json
│   └── *.mp4
└── test_videos/
    └── *.mp4
```

The dataset is not included in this repository. Download the **Deepfake
Detection Challenge** dataset from [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge/data).
Access requires a Kaggle account and acceptance of the competition rules.

## Train and evaluate

From the repository root:

```bash
python -m deepfake_detection.train
```

The script makes a stratified 80/20 split of the labeled sample videos. It extracts frame features, trains the GRU classifier, and evaluates once on the held-out validation split. Labels are encoded as `REAL = 0` and `FAKE = 1`.

Optional settings:

```bash
python -m deepfake_detection.train --data-dir path/to/data --output-dir outputs --epochs 50 --batch-size 64 --max-frames 20
```

The first run can take a long time because every video is decoded and passed through ResNet50.

The best model and run outputs are written to `outputs/`:

- `best_model.keras` — best sequence classifier checkpoint
- `metrics.json` — held-out log loss, accuracy, precision, recall, F1, and ROC AUC
- `evaluation.png` — confusion matrix and ROC curve
- `training_history.png` — training and validation curves
- `run_config.json` — settings and split sizes

## Predict a video

After training, classify a video with:

```bash
python -m deepfake_detection.predict path/to/video.mp4
```

The command prints the predicted class and the model's fake probability.

Optional arguments include `--model outputs/best_model.keras`, `--image-size 224`, and `--max-frames 20`. If training used non-default image or frame settings, pass the same values when predicting.

## Project layout

```text
deepfake_detection/
├── config.py       # Preprocessing and training defaults
├── data.py         # Metadata loading and feature-sequence preparation
├── evaluation.py   # Metrics and evaluation plots
├── features.py     # ResNet50 feature extraction
├── model.py        # GRU classifier and callbacks
├── predict.py      # Single-video prediction command
├── train.py        # Training and validation command
└── video.py        # Video decoding, frame preprocessing, and augmentation
```
