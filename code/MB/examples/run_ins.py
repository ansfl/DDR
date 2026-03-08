import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path to import models
sys.path.append(str(Path(__file__).parent.parent))

from models.strapdown_ins import StrapdownINS

def main():
    # Paths (adjust to your local data)
    imu_file = "path/to/your/imu_data.csv"
    config_file = "../config/config_weinberg.xml"

    # Initialize INS
    ins = StrapdownINS(config_path=config_file)
    
    # Load IMU data
    try:
        imu_data = pd.read_csv(imu_file)
    except FileNotFoundError:
        print(f"IMU file not found: {imu_file}. Please provide a valid CSV.")
        return

    # Process data iteratively
    # Note: INS is sensitive to initial conditions and sensor noise/bias.
    timestamps = imu_data['timestamp'].values
    accels = imu_data[['acc_x', 'acc_y', 'acc_z']].values
    gyros = imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values

    print("Running INS integration...")
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0: continue
        
        ins.update_state(accels[i], gyros[i], dt)

    trajectory = ins.get_trajectory()
    print(f"Processed {len(trajectory)} points.")
    print(f"Final estimated position: {trajectory[-1]}")

if __name__ == "__main__":
    main()
