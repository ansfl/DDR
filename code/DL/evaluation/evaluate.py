import torch
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import os
# Add parent directory to path to allow imports from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xml.etree.ElementTree as ET
from models.mag_model import MagTransformer
from models.dir_model import DirTransformer
from models.resnet_models import VelocityResNet1D
from tqdm import tqdm
import pandas as pd

def load_config(config_path='config.xml'):
    tree = ET.parse(config_path)
    root = tree.getroot()
    config = {
        'input_dim': int(root.find('model/input_dim').text),
        'd_model': int(root.find('model/d_model').text),
        'nhead': int(root.find('model/nhead').text),
        'num_layers': int(root.find('model/num_layers').text),
        'dim_feedforward': int(root.find('model/dim_feedforward').text),
        'dropout': float(root.find('model/dropout').text),
        'device': root.find('training/device').text,
        'output_path': root.find('data/output_path').text,
        'data_path': root.find('data/data_path').text,
        'gps_freq': float(root.find('data/gps_freq').text),
        'imu_freq': float(root.find('data/imu_freq').text),
        'window_size': int(root.find('data/window_size').text),
        'overlap': int(root.find('data/overlap').text),
        'test_trajs': [t.text for t in root.findall('training/test_trajs/traj')]
    }
    return config

def reconstruct_trajectory(mags, d_headings, dt, initial_heading=0):
    # mags: (N,)
    # d_headings: (N,) relative turns in radians
    
    # Integrate headings starting from initial
    headings = initial_heading + np.cumsum(d_headings)
    
    # Transform to Global (0=North, CCW=Positive approach)
    vn = mags * np.cos(headings)
    ve = mags * np.sin(headings)
    
    v = np.stack([vn, ve], axis=1)
    pos = np.zeros((len(v) + 1, 2))
    pos[1:] = np.cumsum(v * dt, axis=0)
    return pos, headings

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.xml')
    parser.add_argument('--mag_model_type', type=str, default='transformer', choices=['transformer', 'resnet'],
                        help='Model type for velocity magnitude (transformer or resnet)')
    parser.add_argument('--dir_model_type', type=str, default='transformer', choices=['transformer', 'resnet'],
                        help='Model type for direction (transformer or resnet)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load Models
    from models.mag_model import MagTransformer
    from models.dir_model import DirTransformer
    from models.resnet_models import VelocityResNet1D, DirectionResNet1D
    
    # Path helper for weights
    def get_weight_path(name):
        # Checks for weight file in current dir or RobotDogPDR root's weights/ folder
        if os.path.exists(name): return name
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        w_path = os.path.join(root, 'weights', name)
        return w_path if os.path.exists(w_path) else name

    # Magnitude Model
    if args.mag_model_type == 'resnet':
        print("Using ResNet1D for Magnitude Prediction")
        mag_model = VelocityResNet1D(in_channels=config['input_dim']).to(device)
        mag_model.load_state_dict(torch.load(get_weight_path('best_mag_resnet.pth'), map_location=device))
    else:
        print("Using Transformer for Magnitude Prediction")
        mag_model = MagTransformer(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        ).to(device)
        mag_model.load_state_dict(torch.load(get_weight_path('best_mag_model.pth'), map_location=device))
    
    mag_model.eval()
    
    # Direction Model
    if args.dir_model_type == 'resnet':
        print("Using ResNet1D for Direction Prediction")
        dir_model = DirectionResNet1D(in_channels=config['input_dim']).to(device)
        w_path = get_weight_path('best_dir_resnet.pth')
        if os.path.exists(w_path):
            dir_model.load_state_dict(torch.load(w_path, map_location=device))
    else:
        print("Using Transformer for Direction Prediction")
        dir_model = DirTransformer(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        ).to(device)
        dir_model.load_state_dict(torch.load(get_weight_path('best_dir_model.pth'), map_location=device))
    dir_model.eval()
    
    # Calculate integration time step from window sliding
    # samples are separated by (window_size - overlap) IMU points
    dt = (config['window_size'] - config['overlap']) / config['imu_freq']
    print(f"Integration dt: {dt:.4f}s")
    
    os.makedirs('eval_results', exist_ok=True)
    
    # Trajectories to evaluate
    trajs = config['test_trajs']
    all_metrics = [] # To store results for summary CSV
    
    for traj_name in trajs:
        pth_file = os.path.join(config['output_path'], f'{traj_name}_test.pth')
        if not os.path.exists(pth_file):
            print(f"Test data for {traj_name} not found at {pth_file}. Skipping.")
            continue
            
        data = torch.load(pth_file, weights_only=False)
        windows = torch.FloatTensor(data['windows']).to(device)
        gt_mags = data['mag']
        gt_dirs = data['dir']
        
        # In this evaluation, to "restore" the global trajectory, 
        # we need the absolute orientation (initial heading/course).
        traj_path = os.path.join(config['data_path'], traj_name)
        
        # Flexible filename search (same as preprocess_data.py)
        gps_options = ['gps.csv', 'gps_converted.csv']
        gps_file = None
        for opt in gps_options:
            path = os.path.join(traj_path, opt)
            if os.path.exists(path):
                gps_file = path
                break
        
        if gps_file is None:
            print(f"Warning: No GPS file found for {traj_name} in {traj_path}. Skipping.")
            continue
            
        gps_df = pd.read_csv(gps_file)
        
        # GPS ground truth for comparison plot
        if 'ned_n' in gps_df.columns:
            n, e = gps_df['ned_n'].values, gps_df['ned_e'].values
        else:
            n, e = gps_df['north'].values, gps_df['east'].values
            
        
        # Inference
        with torch.no_grad():
            if args.mag_model_type == 'resnet':
                # ResNet expects (B, Channels, Length)
                pred_mags = mag_model(windows.transpose(1, 2)).cpu().numpy().flatten()
            else:
                pred_mags = mag_model(windows).cpu().numpy().flatten()
                
            if args.dir_model_type == 'resnet':
                # ResNet outputs [B, 2] -> [cos, sin]
                pred_vectors = dir_model(windows.transpose(1, 2))
                # Convert back to radians
                pred_d_headings = torch.atan2(pred_vectors[:, 1], pred_vectors[:, 0]).cpu().numpy().flatten()
            else:
                pred_d_headings = dir_model(windows).cpu().numpy().flatten()
            
        gt_mags = np.array(gt_mags)
        gt_d_headings = np.array(gt_dirs)
        
        # Stationary Mask - Use Ground Truth magnitude to define when stopped
        # This keeps heading identical across different magnitude model evaluations
        pred_d_headings[gt_mags < 0.1] = 0.0
            
        # Extract overall orientation from GPS for alignment (End-to-End)
        dn_gps = n[-1] - n[0]
        de_gps = e[-1] - e[0]
        gps_overall_h = np.arctan2(de_gps, dn_gps)
        
        # Reconstruct with h0=0 to find the "internal" overall heading
        gt_pos_0, _ = reconstruct_trajectory(gt_mags, gt_d_headings, dt, initial_heading=0)
        dn_gt = gt_pos_0[-1, 0] - gt_pos_0[0, 0]
        de_gt = gt_pos_0[-1, 1] - gt_pos_0[0, 1]
        gt_internal_h = np.arctan2(de_gt, dn_gt)
        
        # Align internal GT angle to GPS direction
        h_aligned = gps_overall_h - gt_internal_h
        pred_h = h_aligned
            
        # Final Reconstruction
        gt_pos, gt_headings = reconstruct_trajectory(gt_mags, gt_d_headings, dt, initial_heading=h_aligned)
        pred_pos, pred_headings = reconstruct_trajectory(pred_mags, pred_d_headings, dt, initial_heading=pred_h)
        
        # Start-align the GPS path for plotting at (0,0)
        gps_n = n - n[0]
        gps_e = e - e[0]
        
        # GPS-derived absolute headings for plot comparison
        if 'roll' in gps_df.columns:
            # Use actual sensor heading if available (it's in degrees in robot_dog_data)
            gps_headings_ref = gps_df['roll'].values 
            print(f"Using 'roll' column as heading reference for {traj_name}")
        else:
            # Fallback: derive from positions (radians) and convert to degrees
            gps_headings_rad = np.arctan2(np.diff(e), np.diff(n))
            gps_headings_rad = np.insert(gps_headings_rad, 0, gps_headings_rad[0])
            gps_headings_ref = np.degrees(np.unwrap(gps_headings_rad))
        
        # Metrics
        # Prediction Error: Distance between Predicted End and GT End
        pred_err = np.linalg.norm(gt_pos[-1] - pred_pos[-1])
        # Simulation Error: Distance between Reconstructed GT End and Actual GPS End
        sim_err = np.linalg.norm(gt_pos[-1] - np.array([gps_n[-1], gps_e[-1]]))
        
        # PRMSE: Positional Root Mean Square Error
        # Calculate Euclidean distance at each point (skip first point since it's 0,0 for both)
        point_errors = np.linalg.norm(gt_pos[1:] - pred_pos[1:], axis=1)
        prmse = np.sqrt(np.mean(point_errors**2))
        
        # Path Length Metrics
        gt_len = np.sum(np.sqrt(np.sum(np.diff(gt_pos, axis=0)**2, axis=1)))
        pred_len = np.sum(np.sqrt(np.sum(np.diff(pred_pos, axis=0)**2, axis=1)))
        len_err = pred_len - gt_len
        len_err_pct = (len_err / gt_len) * 100 if gt_len > 0 else 0
        
        print(f"Trajectory {traj_name}:")
        print(f"  - Prediction Error (Final):  {pred_err:.2f}m")
        print(f"  - PRMSE (Positional Error):  {prmse:.2f}m")
        print(f"  - Simulation Error (vs GPS): {sim_err:.2f}m")
        print(f"  - Length Error: {len_err:.2f}m ({len_err_pct:+.2f}%) [GT: {gt_len:.1f}m, Pred: {pred_len:.1f}m]")
        
        # Store for global summary
        all_metrics.append({
            'trajectory': traj_name,
            'final_error_m': pred_err,
            'prmse_m': prmse,
            'sim_error_m': sim_err,
            'gt_length_m': gt_len,
            'pred_length_m': pred_len,
            'length_error_m': len_err,
            'length_error_pct': len_err_pct
        })
        
        # Plotting Trajectory (Existing)
        plt.figure(figsize=(10, 8))
        plt.plot(gps_e, gps_n, 'k-', alpha=0.3, label='Actual GPS (Reference)')
        plt.plot(gt_pos[:, 1], gt_pos[:, 0], 'g-', label='GT (Reconstructed)')
        plt.plot(pred_pos[:, 1], pred_pos[:, 0], 'r--', label='Predicted')
        plt.scatter(0, 0, c='blue', marker='o', label='Start')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Trajectory Reconstruction: {traj_name}\nPred Error: {pred_err:.2f}m, PRMSE: {prmse:.2f}m')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        
        traj_plot_path = f'eval_results/{traj_name}_reconstruction.png'
        plt.savefig(traj_plot_path)
        print(f"Saved trajectory plot to {traj_plot_path}")
        plt.close()

        # New Plot: Velocity Magnitude comparison
        plt.figure(figsize=(12, 6))
        time_axis = np.arange(len(gt_mags)) * dt
        plt.plot(time_axis, gt_mags, 'g-', alpha=0.6, label='GT')
        plt.plot(time_axis, pred_mags, 'r--', alpha=0.8, label='Predicted Velocity Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity Magnitude (m/s)')
        plt.title(f'Velocity Magnitude Comparison: {traj_name}')
        plt.legend()
        plt.grid(True)
        
        mag_plot_path = f'eval_results/{traj_name}_magnitude.png'
        plt.savefig(mag_plot_path)
        print(f"Saved magnitude plot to {mag_plot_path}")
        plt.close()

        # New Plot: Angle comparison (DEGREES)
        plt.figure(figsize=(12, 6))
        # Convert everything to degrees for plotting
        gt_h_deg = np.degrees(np.unwrap(gt_headings))
        pred_h_deg = np.degrees(np.unwrap(pred_headings))
        
        # Plot GPS reference
        gps_h_unwrapped = np.unwrap(np.radians(gps_headings_ref))
        gps_h_deg = np.degrees(gps_h_unwrapped)
        
        offset = gps_h_deg[0] - gt_h_deg[0]
        gt_h_deg += offset
        pred_h_deg += offset

        gps_t = (np.arange(len(gps_h_deg)) / len(gps_h_deg)) * (len(gt_h_deg) * dt)
        plt.plot(gps_t, gps_h_deg, 'k-', alpha=0.3, label='GPS/Sensor Heading (Reference)')
        
        plt.plot(time_axis, gt_h_deg, 'g-', alpha=0.6, label='GT (Window-Integrated)')
        plt.plot(time_axis, pred_h_deg, 'r--', alpha=0.8, label='Predicted Heading')
        plt.xlabel('Time (s)')
        plt.ylabel('Heading (Degrees)')
        plt.title(f'Heading Comparison: {traj_name}')
        plt.legend()
        plt.grid(True)
        
        angle_plot_path = f'eval_results/{traj_name}_angle.png'
        plt.savefig(angle_plot_path)
        print(f"Saved angle plot to {angle_plot_path}")
        plt.close()

        # Export Trajectories to CSV
        csv_dir = os.path.join('eval_results', 'csv_results')
        os.makedirs(csv_dir, exist_ok=True)

        # GT Trajectory
        gt_df = pd.DataFrame({
            'time': time_axis,
            'north': gt_pos[1:, 0],
            'east': gt_pos[1:, 1],
            'heading_deg': gt_h_deg
        })
        gt_csv_path = os.path.join(csv_dir, f'{traj_name}_gt_reconstructed.csv')
        gt_df.to_csv(gt_csv_path, index=False)
        
        # Predicted Trajectory (Contains error length)
        pred_df = pd.DataFrame({
            'time': time_axis,
            'north': pred_pos[1:, 0],
            'east': pred_pos[1:, 1],
            'heading_deg': pred_h_deg,
            'error_m': point_errors # Point-wise error distance
        })
        pred_csv_path = os.path.join(csv_dir, f'{traj_name}_predicted.csv')
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Saved CSV exports to {csv_dir}:")
        print(f"  - {os.path.basename(gt_csv_path)}")
        print(f"  - {os.path.basename(pred_csv_path)}")

    # Save Global Summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = os.path.join('eval_results', 'evaluation_metrics_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved Global Summary to {summary_path}")

if __name__ == "__main__":
    main()
