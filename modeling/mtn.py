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
            nn.Linear(3420,1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),   
        )
        
        self.phase_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3420, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        
        # Define the fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024, 576),
            nn.ReLU(),
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Upsample(size=(480, 640), mode='bilinear', align_corners=True),
            nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

       
    def forward(self, amplitude_tensor, phase_tensor):
        # Encode the amplitude and phase tensors
        amplitude_features = self.amplitude_encoder(amplitude_tensor)
        phase_features = self.phase_encoder(phase_tensor)
        
        # Concatenate and fuse the features
        fused_features = torch.cat((amplitude_features, phase_features), dim=1)
        fused_features = self.fusion_mlp(fused_features)
        
        # Reshape and process through convolution blocks
        reshaped_features = fused_features.view(-1, 1, 24, 24)
        encoded_features = self.encoder(reshaped_features)
        decoded_features = self.decoder(encoded_features)

        
        
        return decoded_features
