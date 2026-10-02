"""Predict whether a video is real or fake using a trained model."""

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from deepfake_detection.config import TrainingConfig
from deepfake_detection.features import build_feature_extractor, extract_frame_features
from deepfake_detection.video import load_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video to classify.")
    parser.add_argument(
        "--model", type=Path, default=Path("outputs/best_model.keras")
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-frames", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video was not found: {args.video}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Trained model was not found: {args.model}")

    config = TrainingConfig(image_size=args.image_size, max_frames=args.max_frames)
    frames = load_video(
        args.video,
        max_frames=config.max_frames,
        image_size=config.image_size,
    )
    features, mask = extract_frame_features(
        frames,
        extractor=build_feature_extractor(config.image_size),
        max_frames=config.max_frames,
        num_features=config.num_features,
    )
    model = keras.models.load_model(args.model)
    fake_probability = float(
        model.predict([features[None, ...], mask[None, ...]], verbose=0)[0, 0]
    )
    label = "FAKE" if fake_probability >= 0.5 else "REAL"
    print(f"Prediction: {label}")
    print(f"Fake probability: {fake_probability:.4f}")


if __name__ == "__main__":
    main()
