"""
Dataset capture script for Poker Dice Classifier

Captures dice face images using rpicam-tcp-client and saves them organized by class (1-6) into data/raw/<class>/.

USAGE:
    python capture_dataset.py --host IP_RASPBERRY --class 1
    python capture_dataset.py --host IP_RASPBERRY --class 3 --count 100
"""

import argparse
import os
from datetime import datetime

import cv2
from rpicam_tcp_client import CameraClient


def capture_images (host, port, dice_class, count, output_dir):
    """ Captures 'count' images and saves them to data/raw/<dice_class>/."""

    # Build destination folder
    save_dir = os.path.join(output_dir, str(dice_class))
    os.makedirs(save_dir, exist_ok=True)

    # Count existing images to avoid overwriting
    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    print(f"Existing images in class {dice_class}: {existing}")
    print(f"Capturing {count} new images")
    print("Press SPACE to capture | Press Q to quit\n")

    captured = 0

    with CameraClient(
        host=host,
        port=port,
        sharpness=8.0,
        contrast=1.2,
        width=640,
        height=480,
    ) as cam:
        while captured < count:
            frame = cam.get_frame()
            if frame is None:
                print("Connection lost")
                break

            # Show libe preview with capture counter
            display = frame.copy()
            cv2.putText(
                img=display,
                text=f"Class: {dice_class} | Captured: {captured}/{count}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.imshow("Capture Dataset - Poker Dice", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Capture interrupted by user.")
                break
            elif key == ord(" "):
                # Generate unique filename using timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"dice_{dice_class}_{timestamp}.jpg"
                filepath = os.path.join(save_dir, filename)

                cv2.imwrite(filepath, frame)
                captured += 1
                print (f"  Saved: {filename} ({captured}/{count})")
    
    cv2.destroyAllWindows()
    print(f"\nDone! {captured} images saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Capture dice face images for dataset."
    )

    parser.add_argument(
        "--host",
        required=True,
        help="Raspberry Pi IP address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="TCP port"
    )
    parser.add_argument(
        "--class",
        dest="dice_class",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help="Dice face to capture (1-6)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to capture (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory (default:data/raw)"
    )

    args = parser.parse_args()

    capture_images(
        host=args.host,
        port=args.port,
        dice_class=args.dice_class,
        count=args.count,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()