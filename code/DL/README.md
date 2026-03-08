# Robot Dog PDR (Pedestrian Dead Reckoning)

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
├── plotting/           # Visualization scripts
│   ├── plot_comparison_real_dog.py
│   ├── plot_comparison.py
│   └── plot_signals.py
├── utils/              # Data processing utilities
│   ├── preprocess_data.py
│   └── augment_real_dog.py
├── weights/            # Pre-trained model weights
└── README.md
```

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Place your IMU and GPS CSV files in the `dataset/` folder and run the preprocessing script:
```bash
python utils/preprocess_data.py --config config/config.xml
```

### 2. Training
You can train either the Transformer or ResNet models. For example, to train the ResNet magnitude model:
```bash
python training/train_mag_resnet.py --config config/config.xml
```
Weights will be saved to the `weights/` directory.

### 3. Evaluation
To evaluate the models and reconstruct trajectories:
```bash
python evaluation/evaluate.py --config config/config.xml --mag_model_type resnet --dir_model_type resnet
```

### 4. Plotting
To generate comparison plots against real dog data:
```bash
python plotting/plot_comparison_real_dog.py
```

## Note on Execution
It is recommended to run all scripts from the **project root directory** (`RobotDogPDR/`) to ensure all relative paths to modules and configuration files are resolved correctly.
