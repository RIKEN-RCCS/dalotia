#!/usr/bin/env python3
import tf_keras as keras  # use keras version 2 by installing tf-keras
from tf_keras import layers, models, initializers


def residual_block(inputs, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        kernel_initializer=initializers.Constant(0.1),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        kernel_initializer=initializers.Constant(0.2),
    )(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            kernel_initializer=initializers.Constant(0.3),
        )(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)


def build_resnet(input_shape=(16, 16, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(
        16, 3, padding="same", kernel_initializer=initializers.Constant(0.4)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 16)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", kernel_initializer=initializers.GlorotNormal(seed=133)
    )(x)

    return models.Model(inputs, outputs)


if __name__ == "__main__":
    model = build_resnet()
    model.save("tensorflow_model", save_format="tf")
   
    for layer in model.layers:
        print(f"{layer.name}:")
        for weight in layer.weights:
            print(f"  {weight.name} = {weight.numpy()}")