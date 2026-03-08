"""
ResNet1D models for velocity and direction prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """
    1D Residual Block for ResNet.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class VelocityResNet1D(nn.Module):
    """
    ResNet1D for velocity prediction.
    
    Input: [B, 6, L] (IMU window)
    Output: [B, 1] (speed magnitude, non-negative via ReLU)
    """
    
    def __init__(self, in_channels=6, stages=[64, 128, 256, 512], stem_kernel_size=7):
        super(VelocityResNet1D, self).__init__()
        
        # Stem: Conv + BN + ReLU + MaxPool
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stages[0], kernel_size=stem_kernel_size, 
                      stride=1, padding=stem_kernel_size//2, bias=False),
            nn.BatchNorm1d(stages[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual stages
        self.inplanes = stages[0]
        self.stage1 = self._make_stage(stages[0], num_blocks=2, stride=1)
        self.stage2 = self._make_stage(stages[1], num_blocks=2, stride=2)
        self.stage3 = self._make_stage(stages[2], num_blocks=2, stride=2)
        self.stage4 = self._make_stage(stages[3], num_blocks=2, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Regression head
        self.fc = nn.Linear(stages[3], 1)
        self.relu_output = nn.ReLU()  # Enforce non-negative speed
        
        # Initialize weights
        self._init_weights()
    
    def _make_stage(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock1D(self.inplanes, out_channels, stride, downsample))
        self.inplanes = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: [B, 6, L]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        x = self.relu_output(x)
        
        return x

class DirectionResNet1D(VelocityResNet1D):
    """
    ResNet1D for velocity direction prediction (as unit vector).
    
    Input: [B, 6, L] (IMU window)
    Output: [B, 2] (Unit vector [cos, sin])
    """
    def __init__(self, in_channels=6, stages=[64, 128, 256, 512], stem_kernel_size=7):
        super(DirectionResNet1D, self).__init__(in_channels, stages, stem_kernel_size)
        # Output [cos, sin]
        self.fc = nn.Linear(stages[3], 2)
        
    def forward(self, x):
        # x shape: [B, 6, L]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Regression output
        x = self.fc(x)
        
        # Normalize to unit vector
        x = F.normalize(x, p=2, dim=1)
        
        return x
