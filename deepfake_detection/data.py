"""Dataset loading and conversion to padded frame-feature sequences."""

from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from deepfake_detection.config import TrainingConfig
from deepfake_detection.features import extract_frame_features
from deepfake_detection.video import load_video


def load_metadata(metadata_file: Path) -> pd.DataFrame:
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Training metadata was not found: {metadata_file}")

    metadata = pd.read_json(metadata_file).T
    required_columns = {"label"}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    metadata = metadata.loc[metadata["label"].isin(["REAL", "FAKE"])].copy()
    if metadata.empty:
        raise ValueError(f"No REAL or FAKE videos were found in {metadata_file}")
    return metadata


def prepare_videos(
    metadata: pd.DataFrame,
    video_dir: Path,
    extractor: keras.Model,
    config: TrainingConfig,
    augment_real: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    count = len(metadata)
    features = np.zeros(
        (count, config.max_frames, config.num_features), dtype=np.float32
    )
    masks = np.zeros((count, config.max_frames), dtype=bool)
    labels = np.zeros((count,), dtype=np.int32)

    for position, (filename, row) in enumerate(metadata.iterrows()):
        video_path = video_dir / filename
        is_fake = row["label"] == "FAKE"
        frames = load_video(
            video_path,
            max_frames=config.max_frames,
            image_size=config.image_size,
            augment=augment_real and not is_fake,
        )
        features[position], masks[position] = extract_frame_features(
            frames,
            extractor=extractor,
            max_frames=config.max_frames,
            num_features=config.num_features,
        )
        labels[position] = int(is_fake)

    return (features, masks), labels
