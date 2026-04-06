import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def create_dice_model(input_size=(224, 224, 3), num_classes=6):
    # 1. Capa de entrada
    inputs = layers.Input(shape=input_size, name="input_image")

    # 2. Preprocesado específico de MobileNetV2
    x = preprocess_input(inputs)

    # 3. Cargar MobileNetV2 sin la capa final (include_top=False)
    base_model = MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    # 4. Pooling global para reducir dimensiones
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)

    # 5. Capa densa final con 6 clases y softmax
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    # 6. Crear modelo completo
    model = models.Model(inputs=inputs, outputs=outputs, name="dice_classifier")

    return model


if __name__ == "__main__":
    model = create_dice_model()
    model.summary()