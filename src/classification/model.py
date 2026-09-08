"""MobileNetV2 transfer-learning classifier.

Stage (a): ImageNet base frozen, train the head only.
Stage (b): unfreeze the last `fine_tune_layers` of the base at a lower LR.
All hyperparameters come from configs/config.yaml.
"""

import tensorflow as tf


def build_model(num_classes, img_size=224, dropout=0.3):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inputs, outputs, name="oraltwin_classifier")


def unfreeze_top(model, fine_tune_layers):
    """Make the last `fine_tune_layers` layers of the MobileNetV2 base trainable
    (BatchNorm layers stay frozen, standard fine-tuning practice)."""
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base.layers[-fine_tune_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
