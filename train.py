import argparse
import tensorflow as tf

from model_definition import create_dice_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento Poker-Dice con MobileNetV2")

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
        help="Directorio con las imágenes de validación organizadas por clase"
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
        default=20,
        help="Número de épocas de entrenamiento",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Tasa de aprendizaje del optimizador",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="poker_mobilenetv3.h5",
        help="Ruta donde se guardará el modelo entrenado",
    )

    return parser.parse_args()

def create_datasets(args):
    img_size = (args.img_size, args.img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.val_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Prefetch para rendimiento
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds

def build_and_compile_model(args, fine_tune_at=None, learning_rate=None):
    input_shape = (args.img_size, args.img_size, 3)
    num_classes = 6

    model = create_dice_model(
        input_size=input_shape,
        num_classes=num_classes,
        fine_tune_at=fine_tune_at,
        )

    lr = learning_rate if learning_rate is not None else args.learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def main():
    args = parse_args()
    
    # 1.Create datasets
    train_ds, val_ds = create_datasets(args)
    
    # 2.Create and compile model
    model = build_and_compile_model(args)

    # 3.Save best model according to metrics ("val_accuracy")
    checkpoint_cb = ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    # 4.Stops training when validation metrics is not improving for "patience" epochs
    earlystop_cb = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # 5.Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
    )

    # 6.Save trained model
    model.save(args.output_model)
    
    print(f"Path to saved model: {args.output_model}")


if __name__ == "__main__":
    main()

