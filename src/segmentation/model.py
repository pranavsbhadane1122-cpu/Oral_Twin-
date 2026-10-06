"""A modest U-Net, sized for CPU training.

Encoder-decoder with skip connections. Two sigmoid output channels - oral cavity
and lesion - rather than a softmax, because the two overlap: every lesion is
inside the cavity, so the channels are not mutually exclusive.

Depth and width come from configs/config.yaml. The defaults (base 16 filters,
depth 4) are deliberately small: this has to train on a laptop CPU.
"""

import tensorflow as tf

from src.utils.config import load_config


def conv_block(x, filters, name):
    for i in (1, 2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same",
                                   name=f"{name}_conv{i}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = tf.keras.layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def build_unet(img_size=256, channels=2, base_filters=16, depth=4):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    skips = []

    for level in range(depth):
        filters = base_filters * (2 ** level)
        x = conv_block(x, filters, f"enc{level}")
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"pool{level}")(x)

    x = conv_block(x, base_filters * (2 ** depth), "bridge")

    for level in reversed(range(depth)):
        filters = base_filters * (2 ** level)
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                                            name=f"up{level}")(x)
        x = tf.keras.layers.Concatenate(name=f"skip{level}")([x, skips[level]])
        x = conv_block(x, filters, f"dec{level}")

    outputs = tf.keras.layers.Conv2D(channels, 1, activation="sigmoid",
                                     name="mask")(x)
    return tf.keras.Model(inputs, outputs, name="oraltwin_unet")


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Mean Dice over the batch and channels, on soft predictions."""
    axes = [1, 2]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    totals = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    return tf.reduce_mean((2.0 * intersection + smooth) / (totals + smooth))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(bce_weight=0.5):
    """bce_weight * BCE + (1 - bce_weight) * Dice.

    BCE alone collapses on this data: lesions cover a small fraction of the
    frame, so predicting all-zero is already a good cross-entropy. Dice supplies
    the overlap pressure that stops that.
    """
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss(y_true, y_pred):
        return bce_weight * bce(y_true, y_pred) + (1.0 - bce_weight) * dice_loss(
            y_true, y_pred)

    loss.__name__ = "bce_dice"
    return loss


def channel_dice_metric(index, name):
    def metric(y_true, y_pred):
        return dice_coefficient(y_true[..., index:index + 1],
                                y_pred[..., index:index + 1])
    metric.__name__ = name
    return metric


def build_and_compile(cfg=None):
    cfg = cfg or load_config()
    scfg = cfg["segmentation"]
    model = build_unet(scfg["img_size"], len(scfg["channels"]),
                       scfg["base_filters"], scfg["depth"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=scfg["learning_rate"]),
        loss=combined_loss(scfg["bce_weight"]),
        metrics=[channel_dice_metric(0, "dice_cavity"),
                 channel_dice_metric(1, "dice_lesion")],
    )
    return model
