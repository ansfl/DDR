import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.signal import find_peaks
from typing import Tuple, Dict, List

def load_config_xml(config_path: str) -> Dict:
    """Load configuration from XML file."""
    tree = ET.parse(config_path)
    root = tree.getroot()
    
    config = {}
    weinberg = root.find('weinberg')
    if weinberg is not None:
        config['k_constant'] = float(weinberg.find('k_constant').text)
        config['min_peak_height'] = float(weinberg.find('min_peak_height').text)
        config['min_peak_distance'] = int(weinberg.find('min_peak_distance').text)
        config['use_smoothing'] = weinberg.find('use_smoothing').text.lower() == 'true'
        config['window_size'] = int(weinberg.find('window_size').text)
        config['acc_min_threshold'] = float(weinberg.find('acc_min_threshold').text)
        config['acc_max_threshold'] = float(weinberg.find('acc_max_threshold').text)
        config['use_madgwick'] = weinberg.find('use_madgwick').text.lower() == 'true'
        config['madgwick_beta'] = float(weinberg.find('madgwick_beta').text)
        config['mirror_y'] = weinberg.find('mirror_y').text.lower() == 'true'
    
    return config

class MadgwickFilter:
    """Simplified Madgwick filter for heading (yaw) estimation."""
    def __init__(self, beta: float = 0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
    
    def update(self, gyro_z: float, dt: float):
        q_dot = 0.5 * np.array([-self.q[3] * gyro_z, 0, 0, self.q[0] * gyro_z])
        self.q = self.q + q_dot * dt
        norm = np.linalg.norm(self.q)
        if norm > 0: self.q = self.q / norm
    
    def get_heading(self) -> float:
        return np.arctan2(2.0 * (self.q[0] * self.q[3] + self.q[1] * self.q[2]),
                          1.0 - 2.0 * (self.q[2]**2 + self.q[3]**2))

class WeinbergStepDetector:
    """
    Weinberg-based step detection and length estimation from IMU data.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the step detector with configuration.
        
        Args:
            config_path: Path to configuration XML file
        """
        self.config = load_config_xml(config_path)
        self.steps_detected = 0
        self.step_lengths = []
        self.step_times = []
        self.acc_magnitude = None
        self.peaks = None
        
    def calculate_acc_magnitude(self, imu_data: pd.DataFrame) -> np.ndarray:
        """Calculate acceleration magnitude from 3-axis data."""
        acc_mag = np.sqrt(
            imu_data['acc_x']**2 + 
            imu_data['acc_y']**2 + 
            imu_data['acc_z']**2
        )
        return acc_mag.values
    
    def smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average smoothing."""
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal, kernel, mode='same')
    
    def detect_steps(self, imu_data: pd.DataFrame) -> Tuple[int, np.ndarray, np.ndarray]:
        """Detect steps from IMU data using peak detection."""
        self.acc_magnitude = self.calculate_acc_magnitude(imu_data)
        
        if self.config['use_smoothing']:
            self.acc_magnitude = self.smooth_signal(
                self.acc_magnitude, 
                self.config['window_size']
            )
        
        self.peaks, _ = find_peaks(
            self.acc_magnitude,
            height=self.config['min_peak_height'],
            distance=self.config['min_peak_distance']
        )
        
        self.steps_detected = len(self.peaks)
        self.step_times = imu_data['timestamp'].values[self.peaks]
        
        return self.steps_detected, self.peaks, self.acc_magnitude
    
    def calculate_step_lengths(self) -> np.ndarray:
        """Calculate step length using Weinberg formula."""
        if self.peaks is None or len(self.peaks) == 0:
            return np.array([])
        
        K = self.config['k_constant']
        step_lengths = []
        
        for i in range(len(self.peaks)):
            if i == 0:
                start_idx = 0
            else:
                start_idx = (self.peaks[i-1] + self.peaks[i]) // 2
            
            if i == len(self.peaks) - 1:
                end_idx = len(self.acc_magnitude)
            else:
                end_idx = (self.peaks[i] + self.peaks[i+1]) // 2
            
            step_acc = self.acc_magnitude[start_idx:end_idx]
            
            if len(step_acc) > 0:
                a_max = np.max(step_acc)
                a_min = np.min(step_acc)
                step_length = K * np.power(a_max - a_min, 0.25)
                step_lengths.append(step_length)
        
        self.step_lengths = np.array(step_lengths)
        return self.step_lengths

    def calculate_heading(self, imu_data: pd.DataFrame) -> np.ndarray:
        """Calculate heading using Madgwick filter at each detected step."""
        gyro_z = imu_data['gyro_z'].values
        timestamps = imu_data['timestamp'].values
        
        filter = MadgwickFilter(beta=self.config.get('madgwick_beta', 0.1))
        all_headings = np.zeros(len(gyro_z))
        for i in range(1, len(gyro_z)):
            dt = timestamps[i] - timestamps[i-1]
            filter.update(gyro_z[i], dt)
            all_headings[i] = filter.get_heading()
                
        return all_headings[self.peaks] if self.peaks is not None else all_headings

    def build_trajectory(self, headings: np.ndarray) -> np.ndarray:
        """Build 2D trajectory from step lengths and headings."""
        if len(self.step_lengths) == 0 or len(headings) == 0:
            return np.array([[0, 0]])
        
        trajectory = np.zeros((len(self.step_lengths) + 1, 2))
        
        for i in range(min(len(self.step_lengths), len(headings))):
            dn = self.step_lengths[i] * np.cos(headings[i])
            de = self.step_lengths[i] * np.sin(headings[i])
            trajectory[i+1, 0] = trajectory[i, 0] + dn 
            trajectory[i+1, 1] = trajectory[i, 1] + de  
        
        if self.config.get('mirror_y', False):
            trajectory[:, 1] = -trajectory[:, 1]
            
        return trajectory
