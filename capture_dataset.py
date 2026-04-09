"""
Dataset capture script for Poker Dice Classifier

Captures dice face images using rpicam-tcp-client and saves them organized by class (1-6) into data/raw/<class>/.

Configuration is loaded from capture_config.json
The --host argument overrides the host in the config file if provided.

USAGE:
    python capture_dataset.py --class 1
    python capture_dataset.py --class 3 --count 100
    python capture_dataset.py --class 2 --host 192.168.1.50
    python capture_dataset.py --class 1 --config my_config.json

"""

import argparse
import json
import os
from datetime import datetime

import cv2
from rpicam_tcp_client import CameraClient

def load_config(config_path):
    """ Loads configuration from a JSON file. """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Make sure capture_config.json exists in the project root"
        )
    with open(config_path, "r") as f:
        return json.load(f)


def capture_images (host, port, dice_class, count, output_dir, camera_params):
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
        width=camera_params.get("width"),
        height=camera_params.get("height"),
        jpeg_quality=camera_params.get("jpeg_quality"),
        sharpness=camera_params.get("sharpness"),
        contrast=camera_params.get("contrast"),
        brightness=camera_params.get("brightness"),
        saturation=camera_params.get("saturation"),
        exposure_time=camera_params.get("exposure_time"),
        analogue_gain=camera_params.get("analogue_gain")
    ) as cam:
        while captured < count:
            frame = cam.get_frame()
            if frame is None:
                print("Connection lost")
                break

            # Show live preview with capture counter
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
        "--class",
        dest="dice_class",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help="Dice face to capture (1-6)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Raspberry Pi IP address"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of images to capture (default: 100)"
    )
    parser.add_argument(
        "--config",
        default="capture_config.json",
        help="Path to config file (default: capture_config.json)"
    )

    args = parser.parse_args()

    # Load configuration from JSON
    config = load_config(args.config)

    # Command line arguments override config file values
    host = args.host or config["connection"]["host"]
    port = config["connection"]["port"]
    count = args.count or config["capture"]["count"]
    output_dir = config["capture"]["output_dir"]
    camera_params = config["camera"]

    print(f"Config loaded from: {args.config}")
    print(f"Connecting to: {host}:{port}")
    print(f"Camera params: {camera_params}\n")

    capture_images(
        host=host,
        port=port,
        dice_class=args.dice_class,
        count=count,
        output_dir=output_dir,
        camera_params=camera_params,
    )


if __name__ == "__main__":
    main()