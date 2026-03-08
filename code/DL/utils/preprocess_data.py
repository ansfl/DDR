import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch

def load_config(config_path='config/config.xml'):
    if not os.path.exists(config_path):
        # Try relative to the script's folder if run from utils/
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', os.path.basename(config_path))
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            print(f"Warning: {config_path} not found. Using defaults.")
            raise FileNotFoundError(f"Config file {config_path} not found.")

    tree = ET.parse(config_path)
    root = tree.getroot()
    
    # Helper to parse lists
    def parse_list(parent_node):
        if parent_node is None: return []
        return [t.text for t in parent_node.findall('traj')]

    config = {
        'window_size': int(root.find('data/window_size').text),
        'overlap': int(root.find('data/overlap').text),
        'gps_freq': float(root.find('data/gps_freq').text),
        'imu_freq': float(root.find('data/imu_freq').text),
        'data_path': root.find('data/data_path').text,
        'output_path': root.find('data/output_path').text,
        'val_trajs': parse_list(root.find('training/val_trajs')),
        'train_trajs': parse_list(root.find('training/train_trajs')),
        'test_trajs': parse_list(root.find('training/test_trajs'))
    }
    
    # Parse normalization
    norm = root.find('training/normalization')
    if norm is not None:
        config['acc_mean'] = [float(x) for x in norm.find('acc_mean').text.split(',')]
        config['acc_std'] = [float(x) for x in norm.find('acc_std').text.split(',')]
        config['gyro_mean'] = [float(x) for x in norm.find('gyro_mean').text.split(',')]
        config['gyro_std'] = [float(x) for x in norm.find('gyro_std').text.split(',')]
    
    return config

def compute_velocity_labels(gps_df, freq):
    # Compute derivative of NED coordinates
    dt = 1.0 / freq
    
    # Handle naming conventions
    if 'ned_n' in gps_df.columns:
        n, e, d = 'ned_n', 'ned_e', 'ned_d'
    elif 'north' in gps_df.columns:
        n, e, d = 'north', 'east', 'down'
    else:
        raise ValueError(f"GPS dataframe missing North/East columns. Found: {gps_df.columns}")
        
    vn = np.diff(gps_df[n].values) / dt
    ve = np.diff(gps_df[e].values) / dt
    
    # Check if down exists
    if d in gps_df.columns:
        vd = np.diff(gps_df[d].values) / dt
    else:
        vd = np.zeros_like(vn)
    
    # Pad to match original length
    vn = np.append(vn, vn[-1])
    ve = np.append(ve, ve[-1])
    vd = np.append(vd, vd[-1])
    
    # Heading Calculation
    if 'roll' in gps_df.columns: # roll = heading angle in legged robo dataset.
        # User specified that 'roll' in their original data is actually heading.
        heading_deg = gps_df['roll'].values
        heading_rad = np.radians(heading_deg)
        print("Using 'roll' column for heading.")
    else:
        # Fallback for real_dog_data: derive from velocity
        heading_rad = np.arctan2(ve, vn)
        print("Deriving heading from North/East velocity.")
    
    heading_rad = np.unwrap(heading_rad) # Use unwrap for clean derivatives
    
    # Simple moving average smoothing (small window to avoid blurring turns)
    window = 2 if freq > 5 else 1
    if window > 1:
        heading_rad = np.convolve(heading_rad, np.ones(window)/window, mode='same')
    
    d_heading = np.diff(heading_rad) / dt # Radians per second (angular velocity)
    
    # Pad to match original length
    d_heading = np.insert(d_heading, 0, 0.0)
    
    # Magnitude
    mag = np.sqrt(vn**2 + ve**2 + vd**2)
    
    gps_df['v_mag'] = mag
    gps_df['d_heading'] = d_heading # We keep label name for compatibility
    
    return gps_df

def process_trajectory(traj_path, config):
    # Flexible filename search
    gps_options = ['gps.csv', 'gps_converted.csv']
    imu_options = ['imu.csv', 'imu_converted.csv']
    
    gps_file = None
    for opt in gps_options:
        path = os.path.join(traj_path, opt)
        if os.path.exists(path):
            gps_file = path
            break
            
    imu_file = None
    for opt in imu_options:
        path = os.path.join(traj_path, opt)
        if os.path.exists(path):
            imu_file = path
            break
            
    if gps_file is None or imu_file is None:
        print(f"Warning: Missing GPS or IMU file in {traj_path}. Skipping.")
        return None, None, None
        
    gps_df = pd.read_csv(gps_file)
    imu_df = pd.read_csv(imu_file)
    
    # Compute velocity labels at GPS frequency (10Hz)
    gps_df = compute_velocity_labels(gps_df, config['gps_freq'])
    
    # Synchronize IMU and GPS
    # GNSS is 10Hz, IMU is 100Hz.
    # We want a target timestamp for each GPS label that aligns with IMU data.
    # Let's use the closest IMU sample for each GPS label.
    
    imu_times = imu_df['timestamp'].values
    gps_times = gps_df['timestamp'].values
    
    windows = []
    labels_mag = []
    labels_dir = []
    
    window_size = config['window_size']
    
    # Fast window extraction using numpy
    imu_features = imu_df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    
    # Standardization using global config stats
    imu_features[:, 0] = (imu_features[:, 0] - config['acc_mean'][0]) / config['acc_std'][0]
    imu_features[:, 1] = (imu_features[:, 1] - config['acc_mean'][1]) / config['acc_std'][1]
    imu_features[:, 2] = (imu_features[:, 2] - config['acc_mean'][2]) / config['acc_std'][2]
    imu_features[:, 3] = (imu_features[:, 3] - config['gyro_mean'][0]) / config['gyro_std'][0]
    imu_features[:, 4] = (imu_features[:, 4] - config['gyro_mean'][1]) / config['gyro_std'][1]
    imu_features[:, 5] = (imu_features[:, 5] - config['gyro_mean'][2]) / config['gyro_std'][2]
    
    # Sliding window based on overlap
    step = window_size - config['overlap']
    if step < 1: step = 1
    
    # Pre-interpolate labels to IMU frequency
    imu_mag = np.interp(imu_times, gps_times, gps_df['v_mag'].values)
    imu_d_heading = np.interp(imu_times, gps_times, gps_df['d_heading'].values)
    
    for idx_imu in range(window_size, len(imu_features), step):
        start_idx = idx_imu - window_size
        end_idx = idx_imu
        
        if imu_mag[idx_imu] < 0.1: 
            continue
            
        imu_window = imu_features[start_idx:end_idx]
        
        # Window step time for integration in evaluation
        window_step_dt = (config['window_size'] - config['overlap']) / config['imu_freq']
        
        windows.append(imu_window)
        labels_mag.append(imu_mag[idx_imu])
        # Target is turn per window-step
        labels_dir.append(imu_d_heading[idx_imu] * window_step_dt ) 
        
    return np.array(windows), np.array(labels_mag), np.array(labels_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.xml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    os.makedirs(config['output_path'], exist_ok=True)
    
    # Automatically find trajectory folders in data_path
    trajs = [d for d in os.listdir(config['data_path']) if os.path.isdir(os.path.join(config['data_path'], d))]
    
    train_windows, train_mag, train_dir = [], [], []
    val_windows, val_mag, val_dir = [], [], []
    test_data = {} # Per trajectory for reconstruction
    
    for traj in tqdm(trajs, desc="Processing trajectories"):
        traj_path = os.path.join(config['data_path'], traj)
        windows, mag, direction = process_trajectory(traj_path, config)
        
        if traj in config['test_trajs']:
            test_data[traj] = {
                'windows': windows,
                'mag': mag,
                'dir': direction
            }
        elif traj in config['val_trajs']:
            val_windows.extend(windows)
            val_mag.extend(mag)
            val_dir.extend(direction)
        elif not config['train_trajs'] or traj in config['train_trajs']:
            train_windows.extend(windows)
            train_mag.extend(mag)
            train_dir.extend(direction)
            
    print(f"Split Summary:")
    print(f"  Train: {len(train_windows)} windows")
    print(f"  Val:   {len(val_windows)} windows")
    print(f"  Test:  {len(test_data)} trajectories")

    # Save training and validation data
    torch.save({
        'windows': np.array(train_windows),
        'mag': np.array(train_mag),
        'dir': np.array(train_dir)
    }, os.path.join(config['output_path'], 'train.pth'))
    
    torch.save({
        'windows': np.array(val_windows),
        'mag': np.array(val_mag),
        'dir': np.array(val_dir)
    }, os.path.join(config['output_path'], 'val.pth'))
    
    # Save test data separately per trajectory
    for traj, data in test_data.items():
        torch.save(data, os.path.join(config['output_path'], f'{traj}_test.pth'))
        
    print(f"Preprocessing complete. Data saved to {config['output_path']}")

if __name__ == "__main__":
    main()
