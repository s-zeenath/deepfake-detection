"""ResNet feature extraction for sampled video frames."""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_feature_extractor(image_size: int) -> keras.Model:
    backbone = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(image_size, image_size, 3),
    )
    inputs = keras.Input((image_size, image_size, 3))
    preprocessed = keras.applications.resnet50.preprocess_input(inputs)
    features = backbone(preprocessed, training=False)
    return keras.Model(inputs, features, name="resnet50_feature_extractor")


def extract_frame_features(
    frames: np.ndarray,
    extractor: keras.Model,
    max_frames: int,
    num_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((max_frames, num_features), dtype=np.float32)
    mask = np.zeros((max_frames,), dtype=bool)
    frame_count = min(len(frames), max_frames)

    if frame_count:
        extracted = extractor(
            tf.convert_to_tensor(frames[:frame_count]), training=False
        ).numpy()
        features[:frame_count] = extracted
        mask[:frame_count] = True

    return features, mask
