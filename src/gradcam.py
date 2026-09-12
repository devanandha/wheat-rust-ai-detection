import numpy as np
import tensorflow as tf
from PIL import Image


def _find_mobilenet(model: tf.keras.Model) -> tf.keras.Model:
    """
    Find the MobileNetV2 backbone inside the trained classifier.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer

    raise ValueError("Could not find the MobileNetV2 backbone in the model.")


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_index: int | None = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a classifier prediction.

    Parameters
    ----------
    model:
        Trained Wheat Rust classifier.

    image_batch:
        Input image with shape (1, height, width, 3).
        Use the same raw resized pixel values passed to model.predict().

    class_index:
        Class to explain. If None, the model's predicted class is used.

    Returns
    -------
    numpy.ndarray
        Normalised 2D heatmap with values between 0 and 1.
    """

    backbone = _find_mobilenet(model)

    # Deeper 14x14 MobileNetV2 feature map for higher-resolution Grad-CAM.
    last_conv_layer = backbone.get_layer("block_12_add")

    # Model that exposes both the convolutional features
    # and the backbone output.
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv_layer.output, backbone.output],
    )

    # Retrieve the classifier layers that follow MobileNetV2.
    pooling_layer = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    )

    prediction_layer = model.get_layer("prediction")

    image_tensor = tf.cast(image_batch, tf.float32)

    # Apply the same preprocessing used by the classifier.
    processed = tf.keras.applications.mobilenet_v2.preprocess_input(image_tensor)

    with tf.GradientTape() as tape:
        conv_output, backbone_output = backbone_grad_model(
            processed,
            training=False,
        )

        pooled = pooling_layer(backbone_output)

        # Dropout is intentionally skipped during explanation because
        # inference-mode dropout is effectively an identity operation.
        predictions = prediction_layer(pooled)

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_output)

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    # Importance weight for each feature channel.
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * weights,
        axis=-1,
    )

    # Standard Grad-CAM ReLU.
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        heatmap,
    )

    return heatmap.numpy()


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on the original image.
    """

    original = image.convert("RGB")
    width, height = original.size

    heatmap_tensor = tf.convert_to_tensor(
        heatmap[..., np.newaxis],
        dtype=tf.float32,
    )

    heatmap_tensor = tf.image.resize(
        heatmap_tensor,
        (height, width),
    )

    heatmap_resized = heatmap_tensor.numpy().squeeze()

    # Create a simple red-yellow style activation overlay.
    red = np.clip(heatmap_resized * 255, 0, 255)
    green = np.clip(heatmap_resized * 180, 0, 255)
    blue = np.zeros_like(red)

    coloured_heatmap = np.stack(
        [red, green, blue],
        axis=-1,
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(coloured_heatmap)

    return Image.blend(
        original,
        heatmap_image,
        alpha=alpha,
    )
def heatmap_to_image(
    heatmap: np.ndarray,
    size: tuple[int, int],
) -> Image.Image:
    """
    Convert a Grad-CAM heatmap into a high-contrast RGB image.
    """
    width, height = size

    heatmap_tensor = tf.convert_to_tensor(
        heatmap[..., np.newaxis],
        dtype=tf.float32,
    )

    heatmap_tensor = tf.image.resize(
        heatmap_tensor,
        (height, width),
    )

    h = np.clip(heatmap_tensor.numpy().squeeze(), 0.0, 1.0)

    red = np.clip(255 * np.minimum(1.0, 2.0 * h), 0, 255)
    green = np.clip(255 * np.minimum(1.0, 2.0 * (1.0 - np.abs(h - 0.5))), 0, 255)
    blue = np.clip(255 * np.minimum(1.0, 2.0 * (1.0 - h)), 0, 255)

    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)

    return Image.fromarray(rgb)