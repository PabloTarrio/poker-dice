import argparse
import tensorflow as tf

from model_definition import create_dice_model
from train import create_datasets, build_and_compile_model

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Poker-Dice with MobileNetV2")

    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/train",
        help="Directorio con las imágenes de entrenamiento organizadas por clase",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/val",
        help="Directorio con las imágenes de validación organizadas por clase",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Tamaño de imagen (img_size x img_size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch para entrenamiento",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Número de épocas de fine-tuning",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Tasa de aprendizaje para fine-tuning (más pequeña que en el entrenamiento base)",
    )
    parser.add_argument(
        "--fine-tune-at",
        type=int,
        default=100,
        help="Índice de capa de MobileNetV2 desde la que se descongelan capas para fine-tuning",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="models/poker_mobilenetv2_base.keras",
        help="Ruta del modelo previamente entrenado (fase base)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/poker_mobilenetv2_finetuned.keras",
        help="Ruta donde se guardará el modelo tras el fine-tuning",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # 1.Create datasets (reuse train.py function)
    train_ds, val_ds = create_datasets(args)

    # 2.Load pre-trained model
    model = tf.keras.models.load_model(args.base_model_path)
        # 2.1 See how many layers (and their index) the model have.
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

    # 3. Unfreeze last layers from full model
    #       (every layer from fine_tune_at to the end)
    for layer in model.layers[args.fine_tune_at:]:
        layer.trainable = True

    # 4. Compile again with small learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer = optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 5. Re-Train (fine-tuning)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
    )

    # 6. Save fine-tuned model
    model.save(args.output_model)
    print(f"Fine-tuned model saved at: {args.output_model}")


if __name__== "__main__":
    main()