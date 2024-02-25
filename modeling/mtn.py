import torch.nn as nn
import torch
import torch.nn.functional as F
from modeling.build_model import ModalityTranslationNetwork_REGISTRY

__all__ = ["ModalityTranslationNetwork"]

@ModalityTranslationNetwork_REGISTRY.register()
class ModalityTranslationNetwork(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        
        # Define the MLPs for amplitude and phase
        self.amplitude_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3420,2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048), 
            nn.Linear(2048, 1024),
            nn.ReLU(),   
        )
        
        self.phase_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3420, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048), 
            nn.Linear(2048, 1024),
            nn.ReLU(), 
        )
        
        # Define the fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,576),
            nn.ReLU(),
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Upsample(size=(480,640), mode='bilinear', align_corners=True),
            nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

       
    def forward(self, amplitude_tensor, phase_tensor):

        amplitude_features = self.amplitude_encoder(amplitude_tensor)
        phase_features = self.phase_encoder(phase_tensor)
        fused_features = torch.cat((amplitude_features, phase_features), dim=1)
        fused_features = self.fusion_mlp(fused_features)
        reshaped_features = fused_features.view(-1, 1, 24, 24)
        encoded_features = self.encoder(reshaped_features)
        decoded_features = self.decoder(encoded_features)

        
        
        return decoded_features
