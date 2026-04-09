"""
Dataset split script for Poker Dice Classifier.

Splits data/processed/<class>/ into train/val/test sets.

Split ratio: 70% train / 15% val / 15% test

Usage:
    python split_dataset.py
    python split_dataset.py --processed-dir data/processed --output-dir data
    python split_dataset.py --seed 42
"""

import argparse
import os
import random
import shutil


def split_dataset(processed_dir, output_dir, train_ratio, val_ratio, seed):
    """Splits processed images into train/val/test folders."""

    random.seed(seed)

    classes = [str(i) for i in range(1, 7)]
    total_train = total_val = total_test = 0

    for cls in classes:
        src_dir = os.path.join(processed_dir, cls)

        if not os.path.exists(src_dir):
            print(f"Skipping class {cls}: folder not found.")
            continue

        images = sorted([f for f in os.listdir(src_dir) if f.endswith(".jpg")])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dst_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)

            for filename in split_images:
                src = os.path.join(src_dir, filename)
                dst = os.path.join(dst_dir, filename)
                shutil.copy2(src, dst)

        print(
            f"Class {cls}: "
            f"train={len(splits['train'])} | "
            f"val={len(splits['val'])} | "
            f"test={len(splits['test'])}"
        )

        total_train += len(splits["train"])
        total_val += len(splits["val"])
        total_test += len(splits["test"])

    print(f"\nDone!")
    print(f"  Total train : {total_train}")
    print(f"  Total val   : {total_val}")
    print(f"  Total test  : {total_test}")
    print(f"  TOTAL       : {total_train + total_val + total_test}")


def main():
    parser = argparse.ArgumentParser(
        description="Split processed dataset into train/val/test."
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Processed images directory (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output base directory (default: data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    train_ratio = 0.70
    val_ratio = 0.15

    print(f"Processed dir : {args.processed_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Split         : {train_ratio}/{val_ratio}/{round(1-train_ratio-val_ratio, 2)}")
    print(f"Seed          : {args.seed}\n")

    split_dataset(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()