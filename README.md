# Poker Dice Classifier

Neural network to recognize poker dice comnications (1-6 faces) using 5 dice.

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

## Repository structure
```text
|____ model_definition.py          # CNN architecture (MobileNetV2 + Dense(6))
|____ README.md
|____ requirement.txt              # Dependencies
|____ notebooks/                   # Jupyter noteboks for experiments
```

## Quick Starts

```bash
# Clone the repo
git clone https://github.com/PabloTarrio/poker-dice.git
cd poker-dice

# Setup enviroment
python -m venv .venv
source -venv/bin/activate  # Linux/Mac
source .venv/Scripts/activate # Windows
pip install -r requirements.txt

# Test the model arquitecture
python model-definitition.py
```

## Roadmap

- [X] Define CNN architecture (MobileNetV2 transfer learning)
- [ ] Capture dataset (dice images 1-6)
- [ ] Train and evaluate model
- [ ] Implement poker hand logic
- [ ] Integrate with rpicam-tcp-client
- [ ] Real-time inference demo

## Related Projects

 [rpicam-tcp-client](https://github.com/PabloTarrio/rpicam-tcp-client): Camera streaming library