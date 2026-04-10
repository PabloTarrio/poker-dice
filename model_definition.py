import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def create_dice_model(input_size=(224, 224, 3), num_classes=6, fine_tune_at=None):
    # 1. Entry Layer
    inputs = layers.Input(shape=input_size, name="input_image")

    # 2. Data Augmentation (just for training)
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation"
    )

    x = data_augmentation(inputs)

    # 3. MobileNetV2 specific pre-processing
    x = preprocess_input(x)

    # 4. Load MobileNetV2 without final layer (include_top=False)
    base_model = MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights="imagenet"
    )
    # 5. Fine tuning frosted by default
    base_model.trainable = False
    # 5.1 If we want fine-tuning, we defrost from a certain layer. 
    if fine_tune_at is not None:
        for layer in base_model.layers(fine_tune_at):
            layer.trainable = True

    # 6. Global pooling to reduce dimensions
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)

    # 7. Dense final layer with 6 classes and softmax 
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    # 8. Create full model
    model = models.Model(inputs=inputs, outputs=outputs, name="dice_classifier")

    return model


if __name__ == "__main__":
    model = create_dice_model()
    model.summary()