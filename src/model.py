import tensorflow as tf

from .data import IMAGE_SIZE


def _build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def build_classifier(
    num_classes: int = 3,
    architecture: str = "mobilenetv2",
) -> tf.keras.Model:

    architecture = architecture.lower()

    if architecture == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    elif architecture == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    elif architecture == "resnet50":
        base = tf.keras.applications.ResNet50(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input

    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            "Choose from mobilenetv2, efficientnetb0, or resnet50."
        )

    base.trainable = False

    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="image",
    )

    x = _build_augmentation()(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="prediction",
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name=f"wheat_rust_{architecture}",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model