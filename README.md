# Poker Dice Classifier

Neural network to classify individual poker dice faces (1-6) using Transfer Learning, and determine the resulting poker hand using Python logic.

## Project Goals:

1. **Classify individual dice faces** (1, 2, 3, 4, 5, 6) using CNN.

2. **Combine 5 dice results** with Python logic to determine poker hands.

3. **Integrate with rpicamp-tcp-client** for real-time capture from Raspberry Pi.

## Arquitecture

RPi Camera -> rpicam-tcp-client -> Image (224x224)
|
MobileNetV2 (ImageNet weigths)
|
GlobalAveragePooling2D
|
Dense (6, softmax) -> [0.005, 0.92, 0.01, 0.01, 0.01, 0.00]
|
Dice value: 2
|
5 Dice->-> "Full House"

### Model Summary

| Layer | Output Shape | Params |
|---|---|---|
| InputLayer | (None, 224, 224, 3) | 0 |
| MobileNetV2 (frozen) | (None, 7, 7, 1280) | 2,257,984 |
| GlobalAveragePooling2D | (None, 1280) | 0 |
| Dense(6, softmax) | (None, 6) | 7,686 |

- **Total params**: 2,265,670
- **Trainable params**: 7,686 ← only these are updated during training
- **Non-trainable params**: 2,257,984

## Repository structure
```text
poker-dice/
|___ model_definition.py           # CNN architecture (MobileNetV2 + Dense(6))
|___ capture_dataset.py            # Script to capture dice images from Raspberry Pi
|___ capture_config.json           # Camera and capture configuration (copy and edit)
|___ requirements.txt               # Exact dependencies versions
|___ README.md
|___ data/
| |___ raw/                        # Original dice images captured from camera
| |__|___ 1/
| |__|___ 2/
| |__|___ 3/
| |__|___ 4/
| |__|___ 5/
| |__|___ 6/
| |___ processed/                  # Preprocessed images ready for training
|___ models/                       # Saved model weights after training
|___ notebooks/                    # Jupyter noteboks for experiments
```
## Setup

```bash
# Clone the repo
git clone https://github.com/PabloTarrio/poker-dice.git
cd poker-dice

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Test the model architecture
python model_definition.py
```

## Dataset Capture

Images are captured using `capture_dataset.py` with a Raspberry Pi Camera via `rpicam-tcp-client`

### 1.Configure the camera

Copy and edit `capture_config.json` with your Raspberry Pi and preferred camera parameters:

```json
{
    "connection": {
        "host": "192.168.1.XX",
        "port": 5001
    },
    "capture": {
        "count": 100,
        "output_dir": "data/raw"
    },
    "camera": {
        "width": 640,
        "height": 480,
        "jpeg_quality": 95,
        "sharpness": 8.0,
        "contrast": 1.2,
        "brightness": 0.0,
        "saturation": 1.0,
        "exposure_time": null,
        "analogue_gain": null
    }
}
```

### Camera Parameters

| Parameter | Range | Default | Description |
|---|---|---|---|
| `jpeg_quality` | 0–100 | 95 | JPEG compression quality |
| `sharpness` | 0–16 | 8.0 | Image sharpness |
| `contrast` | 0–32 | 1.2 | Image contrast |
| `brightness` | -1.0–1.0 | 0.0 | Image brightness |
| `saturation` | 0–32 | 1.0 | Color saturation |
| `exposure_time` | 114–694267 µs | null | null = auto |
| `analogue_gain` | 1.0–16.0 | null | null = auto |

### 2. Capture images for each dice face

```bash
python capture_dataset.py --class 1
python capture_dataset.py --class 2
python capture_dataset.py --class 3
python capture_dataset.py --class 4
python capture_dataset.py --class 5
python capture_dataset.py --class 6
```

Press `SPACE` to capture each image. Press **Q** to quit.

### 3. Verify dataset

```bash
find data/raw -name "*.jpg" | wc -l # Should return 600
```

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| tensorflow | 2.18.0 | Deep learning framework |
| opencv-python | 4.13.0.92 | Image processing |
| numpy | 2.0.2 | Numerical computing |
| rpicam-tcp-client | 1.0.0 | Raspberry Pi Camera TCP streaming |

## Roadmap

- [x] Define CNN architecture (MobileNetV2 + Transfer Learning)
- [x] Create project folder structure
- [x] Add requirements.txt
- [x] Implement dataset capture script (capture_dataset.py)
- [x] Add configurable camera parameters (capture_config.json)
- [ ] Capture full dataset (dice images, classes 1–6) (100 images x 6 classes = 600 total)
- [ ] Preprocess and augment images
- [ ] Train and evaluate model
- [ ] Implement poker hand logic (pairs, full house, etc.)
- [ ] Integrate with rpicam-tcp-client for real-time inference

## Related Projects

- [rpicam-tcp-client](https://github.com/PabloTarrio/rpicam-tcp-client) — TCP camera streaming library for Raspberry Pi