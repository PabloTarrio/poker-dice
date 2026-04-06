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
|___ requirement.txt               # Exact dependencies versions
|___ README.md
|___ data/
|     |___ raw/                    # Original dice images captured from camera
|     |___ processed/              # Preprocessed images ready for training
|___ models/                       # Saved model weights after training
|___ notebooks/                    # Jupyter noteboks for experiments
```

## Quick Starts

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

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| tensorflow | 2.18.0 | Deep learning framework |
| opencv-python | 4.13.0.92 | Image processing |
| numpy | 2.0.2 | Numerical computing |

## Roadmap

- [x] Define CNN architecture (MobileNetV2 + Transfer Learning)
- [x] Create project folder structure
- [x] Add requirements.txt
- [ ] Capture dataset (dice images, classes 1–6)
- [ ] Preprocess and augment images
- [ ] Train and evaluate model
- [ ] Implement poker hand logic (pairs, full house, etc.)
- [ ] Integrate with rpicam-tcp-client for real-time inference

## Related Projects

- [rpicam-tcp-client](https://github.com/PabloTarrio/rpicam-tcp-client) — TCP camera streaming library for Raspberry Pi