import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xml.etree.ElementTree as ET
import os
import numpy as np
from tqdm import tqdm
import sys
import os
# Add parent directory to path to allow imports from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_models import DirectionResNet1D

def load_config(config_path='config/config.xml'):
    if not os.path.exists(config_path):
        # Try relative to the script's folder if run from training/
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', os.path.basename(config_path))
        if os.path.exists(alt_path):
            config_path = alt_path
            
    tree = ET.parse(config_path)
    root = tree.getroot()
    config = {
        'input_dim': int(root.find('model/input_dim').text),
        'epochs': int(root.find('training/epochs').text),
        'batch_size': int(root.find('training/batch_size').text),
        'learning_rate': float(root.find('training/learning_rate').text),
        'device': root.find('training/device').text,
        'output_path': root.find('data/output_path').text
    }
    return config

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.xml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_pth = os.path.join(config['output_path'], 'train.pth')
    val_pth = os.path.join(config['output_path'], 'val.pth')
    
    if not os.path.exists(train_pth) or not os.path.exists(val_pth):
        print(f"Error: Processed data not found. Run preprocess_data.py first.")
        return

    train_data = torch.load(train_pth, weights_only=False)
    val_data = torch.load(val_pth, weights_only=False)
    
    train_dataset = TensorDataset(torch.FloatTensor(train_data['windows']), torch.FloatTensor(train_data['dir']))
    val_dataset = TensorDataset(torch.FloatTensor(val_data['windows']), torch.FloatTensor(val_data['dir']))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = DirectionResNet1D(in_channels=config['input_dim']).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    
    def cosine_loss(pred, target_angle):
        # pred: [B, 2] (unit vector [cos, sin])
        # target_angle: [B] (radians)
        
        # Convert target angle to unit vector [cos, sin]
        target_v = torch.stack([torch.cos(target_angle), torch.sin(target_angle)], dim=1)
        
        # Dot product of unit vectors is cos(theta)
        cos_theta = torch.sum(pred * target_v, dim=1)
        
        # Loss = 1 - cos(theta)
        return torch.mean(1 - cos_theta)

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            x, y = x.to(device), y.to(device)
            
            # ResNet expects [B, Channels, Length]
            x = x.transpose(1, 2)
            
            # noise = torch.randn_like(x) * 0.01
            # x = x + noise
            
            optimizer.zero_grad()
            pred = model(x)
            
            loss = cosine_loss(pred, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                x, y = x.to(device), y.to(device)
                x = x.transpose(1, 2)
                pred = model(x)
                loss = cosine_loss(pred, y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'best_dir_resnet.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best ResNet direction model (Loss: {best_val_loss:.6f}) to weights/")

if __name__ == "__main__":
    main()
