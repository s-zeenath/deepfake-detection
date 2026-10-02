"""Shared defaults for video preprocessing and model training."""

from dataclasses import dataclass
from pathlib import Path


IMAGE_SIZE = 224
MAX_FRAMES = 20
NUM_FEATURES = 2048
RANDOM_SEED = 42


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = IMAGE_SIZE
    max_frames: int = MAX_FRAMES
    num_features: int = NUM_FEATURES
    batch_size: int = 64
    epochs: int = 50
    random_seed: int = RANDOM_SEED


def metadata_path(data_dir: Path) -> Path:
    return data_dir / "train_sample_videos" / "metadata.json"
