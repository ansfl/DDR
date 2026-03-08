# Robot Dog PDR

This project implements a Pedestrian Dead Reckoning (PDR) system for a robot dog using Deep Learning models (Transformer and ResNet) to predict velocity magnitude and change in heading from IMU data.

## Project Structure

```text
DL/
├── config/           
│   ├── config.xml      # Hyperparameters for robot dog data
│   └── config_real.xml # Hyperparameters for real dog data 
├── models/             # Neural network architectures
│   ├── mag_model.py   
│   ├── dir_model.py    # Transformer for direction
│   └── resnet_models.py # ResNet1D 
├── training/           # Training 
│   ├── train_mag_resnet.py
│   ├── train_dir.py
│   └── train_dir_resnet.py
├── evaluation/         # Evaluation
│   └── evaluate.py
├── utils/              # Data processing 
│   ├── preprocess_data.py
├── weights/           
└── README.md
```
