"""
Dataset preprocessing script for Poker Dice Classifier.

For each image in data/raw/<class>/:
    1. Detects the dice automatically using edge detection (Canny)
    2. Crops the detected area with a small margin
    3. Resizes to 224x224 pixels (MobileNetV2 input size)
    4. Saves to data/processed/<class>/

Usage:
    python preprocess_dataset.py
    python preprocess_dataset.py --input-dir data/raw --output-dir data/processed
    python preprocess_dataset.py --margin 20 --preview
"""

import argparse
import os

import cv2
import numpy as np


def detect_dice_bbox(image, margin=15):
    """
    Detects the dice bounding box using two strategies:
    1. Primary: color detection (beige/cream dice on pink/white background)
    2. Fallback: edge detection (Canny)

    Args:
        image: BGR image as numpy array.
        margin: extra pixels around the detected dice (default 15).

    Returns:
        Tuple (x, y, w, h) of the bounding box, or None if not detected.
    """
    h_img, w_img = image.shape[:2]

    # --- Strategy 1: color detection (beige/cream) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_beige = np.array([10, 20, 150])
    upper_beige = np.array([40, 120, 255])
    mask = cv2.inRange(hsv, lower_beige, upper_beige)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(w_img - x, w + 2 * margin)
            h = min(h_img - y, h + 2 * margin)
            return x, y, w, h

    # --- Strategy 2: fallback with Canny ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    kernel2 = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel2, iterations=2)

    contours2, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours2:
        return None

    largest2 = max(contours2, key=cv2.contourArea)
    if cv2.contourArea(largest2) < 1000:
        return None

    x, y, w, h = cv2.boundingRect(largest2)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(w_img - x, w + 2 * margin)
    h = min(h_img - y, h + 2 * margin)
    return x, y, w, h


def preprocess_image(image, target_size=(224, 224), margin=15):
    """
    Detects the dice, crops it and resizes to target_size.

    Args:
        image: BGR image as numpy array.
        target_size: output size tuple (width, height).
        margin: extra pixels around the detected dice.

    Returns:
        Preprocessed image, or None if dice not detected.
    """
    bbox = detect_dice_bbox(image, margin=margin)

    if bbox is None:
        return None

    x, y, w, h = bbox
    cropped = image[y:y + h, x:x + w]
    resized = cv2.resize(cropped, target_size)

    return resized


def preprocess_dataset(input_dir, output_dir, margin, preview):
    """Processes all images in input_dir and saves them to output_dir."""

    classes = [str(i) for i in range(1, 7)]
    total_ok = 0
    total_failed = 0

    for cls in classes:
        input_cls_dir = os.path.join(input_dir, cls)
        output_cls_dir = os.path.join(output_dir, cls)

        if not os.path.exists(input_cls_dir):
            print(f"Skipping class {cls}: folder not found.")
            continue

        os.makedirs(output_cls_dir, exist_ok=True)

        images = [f for f in os.listdir(input_cls_dir) if f.endswith(".jpg")]
        print(f"\nClass {cls}: processing {len(images)} images...")

        ok = 0
        failed = 0

        for filename in images:
            input_path = os.path.join(input_cls_dir, filename)
            image = cv2.imread(input_path)

            if image is None:
                print(f"  ERROR reading: {filename}")
                failed += 1
                continue

            processed = preprocess_image(image, margin=margin)

            if processed is None:
                print(f"  WARNING: dice not detected in {filename}")
                failed += 1
                continue

            # Show preview if requested
            if preview:
                cv2.imshow(f"Class {cls} - Preview", processed)
                key = cv2.waitKey(200) & 0xFF
                if key == ord("q"):
                    print("Preview interrupted by user.")
                    cv2.destroyAllWindows()
                    return

            output_path = os.path.join(output_cls_dir, filename)
            cv2.imwrite(output_path, processed)
            ok += 1

        print(f"  OK: {ok} | Failed: {failed}")
        total_ok += ok
        total_failed += failed

    cv2.destroyAllWindows()
    print(f"\nDone! Total OK: {total_ok} | Total failed: {total_failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dice images for training."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Input directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=15,
        help="Margin in pixels around detected dice (default: 15)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a preview of each processed image",
    )
    args = parser.parse_args()

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Margin: {args.margin}px")
    print(f"Preview: {args.preview}\n")

    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        margin=args.margin,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()