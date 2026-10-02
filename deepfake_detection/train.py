"""Train and evaluate the CNN-GRU deepfake classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from deepfake_detection.config import TrainingConfig, metadata_path
from deepfake_detection.data import load_metadata, prepare_videos
from deepfake_detection.evaluation import evaluate_predictions, save_evaluation
from deepfake_detection.features import build_feature_extractor
from deepfake_detection.model import build_sequence_model, training_callbacks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/deepfake-detection-challenge"),
        help="Directory containing train_sample_videos/metadata.json and its videos.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_frames < 1:
        raise ValueError("epochs, batch size, and max frames must be positive.")

    config = TrainingConfig(
        image_size=args.image_size,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    metadata = load_metadata(metadata_path(args.data_dir))
    train_metadata, validation_metadata = train_test_split(
        metadata,
        test_size=0.2,
        random_state=config.random_seed,
        stratify=metadata["label"],
    )

    extractor = build_feature_extractor(config.image_size)
    train_data, train_labels = prepare_videos(
        train_metadata,
        args.data_dir / "train_sample_videos",
        extractor,
        config,
        augment_real=True,
    )
    validation_data, validation_labels = prepare_videos(
        validation_metadata,
        args.data_dir / "train_sample_videos",
        extractor,
        config,
    )

    model = build_sequence_model(config.max_frames, config.num_features)
    history = model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_data=(
            [validation_data[0], validation_data[1]],
            validation_labels,
        ),
        callbacks=training_callbacks(args.output_dir),
        epochs=config.epochs,
        batch_size=config.batch_size,
    )

    probabilities = model.predict(
        [validation_data[0], validation_data[1]], verbose=0
    ).reshape(-1)
    metrics = evaluate_predictions(validation_labels, probabilities)
    save_evaluation(
        validation_labels, probabilities, metrics, history, args.output_dir
    )
    with (args.output_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(
            {
                **config.__dict__,
                "training_samples": len(train_metadata),
                "validation_samples": len(validation_metadata),
                "class_mapping": {"REAL": 0, "FAKE": 1},
            },
            config_file,
            indent=2,
        )

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
