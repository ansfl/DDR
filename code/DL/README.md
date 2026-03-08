# Robot Dog PDR

This project implements a Pedestrian Dead Reckoning (PDR) system for a robot dog using Deep Learning models (Transformer and ResNet) to predict velocity magnitude and change in heading from IMU data.

## Project Structure

```text
RobotDogPDR/
├── config/             # Configuration files
│   ├── config.xml      # Hyperparameters and data paths
│   └── config_real.xml # Config for real dog data comparison
├── models/             # Neural network architectures
│   ├── mag_model.py    # Transformer for velocity magnitude
│   ├── dir_model.py    # Transformer for direction
│   └── resnet_models.py # ResNet1D architectures
├── training/           # Training scripts
│   ├── train_mag_resnet.py
│   ├── train_dir.py
│   └── train_dir_resnet.py
├── evaluation/         # Evaluation and metrics
│   └── evaluate.py
├── utils/              # Data processing utilities
│   ├── preprocess_data.py
├── weights/           
└── README.md
```
