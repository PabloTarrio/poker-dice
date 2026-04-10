import argparse
import tensorflow as tf

from train import create_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Poker-Dice model on test set")

    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test",
        help="Directorio con las imagenes de test organizadas por clase",
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
        help="Tamaño de batch para evaluación",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/poker_mobilenetv2_finetuned.keras",
        help="Ruta del modelo entrenado a evaluar",
    )

    return parser.parse_args()

def create_test_dataset(args):
    img_size = (args.img_size, args.img_size)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.test_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    test_ds = test_ds.prefetch(buffer_size=autotune)
    
    return test_ds

def main():
    args = parse_args()

    # 1.Create test dataset
    test_ds = create_test_dataset(args)

    # 2.Load trained model
    model = tf.keras.models.load_model(args.model_path)

    # 3.Evaluate test
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
