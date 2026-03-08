import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, Dict

def load_ins_config_xml(config_path: str) -> Dict:
    """Load INS configuration from XML file."""
    try:
        tree = ET.parse(config_path)
        root = tree.getroot()
        config = {}
        ins = root.find('ins')
        if ins is not None:
            config['gravity'] = float(ins.find('gravity').text)
        else:
            config['gravity'] = 9.81
        return config
    except:
        return {'gravity': 9.81}

class StrapdownINS:
    """
    Strapdown Inertial Navigation System implementation.
    Integrates 6-DOF IMU data (accelerometer + gyroscope) to estimate
    position trajectory in the navigation frame.
    """
    
    def __init__(self, config_path: str = None, gravity: float = 9.81):
        """
        Initialize the INS.
        
        Args:
            config_path: Path to configuration XML file (optional)
            gravity: Gravity magnitude in m/s² (default: 9.81, overridden by config if provided)
        """
        if config_path:
            config = load_ins_config_xml(config_path)
            self.gravity = config.get('gravity', gravity)
        else:
            self.gravity = gravity
        
        # State variables
        self.quaternion = None  # [w, x, y, z]
        self.velocity = None    # [vn, ve, vd] in nav frame
        self.position = None    # [north, east, down] in nav frame
        
        # Results storage
        self.positions_history = []
        self.velocities_history = []
        self.attitudes_history = []
        
    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix (body to navigation frame)."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R
    
    def update_quaternion(self, q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Update quaternion using gyroscope measurements."""
        wx, wy, wz = gyro
        omega_matrix = np.array([
            [0,   -wx,  -wy,  -wz],
            [wx,   0,    wz,  -wy],
            [wy,  -wz,   0,    wx],
            [wz,   wy,  -wx,   0]
        ])
        q_dot = 0.5 * omega_matrix @ q
        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)
        return q_new
    
    def update_state(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """Perform one step of INS integration."""
        if self.quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            self.velocity = np.zeros(3)
            self.position = np.zeros(3)
            
        # 1. Update attitude
        self.quaternion = self.update_quaternion(self.quaternion, gyro, dt)
        R_b_n = self.quaternion_to_rotation_matrix(self.quaternion)
        
        # 2. Transform body acceleration to navigation frame
        acc_n = R_b_n @ acc
        
        # 3. Remove gravity (assuming Z-down navigation frame)
        acc_n[2] -= self.gravity
        
        # 4. Integrate velocity
        self.velocity += acc_n * dt
        
        # 5. Integrate position
        self.position += self.velocity * dt
        
        # Store results
        self.positions_history.append(self.position.copy())
        self.velocities_history.append(self.velocity.copy())
        self.attitudes_history.append(self.quaternion.copy())

    def get_trajectory(self) -> np.ndarray:
        """Return the estimated position trajectory."""
        return np.array(self.positions_history)
