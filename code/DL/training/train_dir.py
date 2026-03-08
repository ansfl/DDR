import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import sys
import os
# Add parent directory to path to allow imports from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dir_model import DirTransformer

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
        'd_model': int(root.find('model/d_model').text),
        'nhead': int(root.find('model/nhead').text),
        'num_layers': int(root.find('model/num_layers').text),
        'dim_feedforward': int(root.find('model/dim_feedforward').text),
        'dropout': float(root.find('model/dropout').text),
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
    
    # Load data
    train_data = torch.load(os.path.join(config['output_path'], 'train.pth'), weights_only=False)
    val_data = torch.load(os.path.join(config['output_path'], 'val.pth'), weights_only=False)
    
    train_dataset = TensorDataset(torch.FloatTensor(train_data['windows']), torch.FloatTensor(train_data['dir']))
    val_dataset = TensorDataset(torch.FloatTensor(val_data['windows']), torch.FloatTensor(val_data['dir']))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = DirTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    # Delta-Heading is a scalar, use MSE loss
    criterion = nn.MSELoss()
    #criterion = nn.HuberLoss(delta=0.1)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            x, y = x.to(device), y.to(device)
            
            # if model.training:
            #     noise = torch.randn_like(x) * 0.01
            #     x = x + noise
                
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y.unsqueeze(1))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'best_dir_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best direction model to weights/")

if __name__ == "__main__":
    main()
