from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EXPECTED_CLASSES = ["Brown_rust", "Healthy", "Yellow_rust"]


def load_split(
    directory: Path,
    *,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
    return_paths: bool = False,
):
    """Load one class-folder image split with deterministic class ordering."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=EXPECTED_CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )
    file_paths = list(dataset.file_paths)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if return_paths:
        return dataset, file_paths

    return dataset


def load_train_and_validation(data_dir: Path, batch_size: int = BATCH_SIZE):
    train = load_split(data_dir / "train", shuffle=True, batch_size=batch_size)
    validation = load_split(data_dir / "val", shuffle=False, batch_size=batch_size)
    return train, validation

