"""Video frame loading and augmentation."""

from pathlib import Path

import cv2
import numpy as np


def crop_center_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    start_x = (width - side) // 2
    start_y = (height - side) // 2
    return frame[start_y : start_y + side, start_x : start_x + side]


def augment_frames(frames: np.ndarray) -> np.ndarray:
    augmented = []
    for frame in frames:
        image = cv2.flip(frame, 1) if np.random.random() < 0.5 else frame.copy()
        height, width = image.shape[:2]
        angle = np.random.uniform(-20, 20)
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(
            image,
            rotation,
            (width, height),
            borderMode=cv2.BORDER_REFLECT_101,
        )

        noise_scale = np.random.uniform(0, 0.05 * 255)
        noise = np.random.normal(0, noise_scale, image.shape)
        image = image.astype(np.float32) + noise
        image *= np.random.uniform(0.8, 1.2)
        image = (image - 127.5) * np.random.uniform(0.8, 1.2) + 127.5
        augmented.append(np.clip(image, 0, 255).astype(np.uint8))

    return np.asarray(augmented)


def load_video(
    path: Path,
    max_frames: int,
    image_size: int,
    augment: bool = False,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")

    frames = []
    try:
        while len(frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (image_size, image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {path}")

    result = np.asarray(frames, dtype=np.uint8)
    return augment_frames(result) if augment else result
