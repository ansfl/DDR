# Dog Navigation Algorithms

This repository contains standalone implementations of navigation algorithms for quadrupeds (robot dogs) or real dogs, focusing on IMU-based trajectory estimation.

## Algorithms

### 1. Weinberg Step Length Model
The Weinberg approach estimates step length using the vertical acceleration range:
`step_length = K * (a_max - a_min)^(1/4)`
- **`models/weinberg.py`**: Core algorithm, step detection, and trajectory building.
- **`examples/run_weinberg.py`**: Basic usage example.

### 2. Strapdown INS
A standard 6-DOF Strapdown Inertial Navigation System that integrates accelerometer and gyroscope data to estimate 3D position and orientation.
- **`models/strapdown_ins.py`**: Quaternion-based attitude estimation and double integration for position.
- **`examples/run_ins.py`**: Basic usage example.

## Configuration
The algorithms use an XML-based configuration file (`config/config.xml`) to manage parameters like the Weinberg constant `K`, peak detection thresholds, and filtering constants.

## Installation

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements
- Python 3.7+
- NumPy
- Pandas
- SciPy
