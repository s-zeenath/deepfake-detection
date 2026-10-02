"""Sequence classifier and training callbacks."""

from pathlib import Path

from tensorflow import keras
from tensorflow.keras import regularizers


def build_sequence_model(max_frames: int, num_features: int) -> keras.Model:
    frame_features = keras.Input((max_frames, num_features), name="frame_features")
    frame_mask = keras.Input((max_frames,), dtype="bool", name="frame_mask")

    sequence = keras.layers.GRU(
        16,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(0.01),
    )(frame_features, mask=frame_mask)
    sequence = keras.layers.BatchNormalization()(sequence)
    sequence = keras.layers.Dropout(0.4)(sequence)
    sequence = keras.layers.GRU(8)(sequence, mask=frame_mask)
    sequence = keras.layers.Dropout(0.4)(sequence)
    sequence = keras.layers.Dense(
        8, activation="relu", kernel_regularizer=regularizers.l2(0.01)
    )(sequence)
    prediction = keras.layers.Dense(
        1, activation="sigmoid", name="fake_probability"
    )(sequence)

    model = keras.Model([frame_features, frame_mask], prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def training_callbacks(output_dir: Path) -> list[keras.callbacks.Callback]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
