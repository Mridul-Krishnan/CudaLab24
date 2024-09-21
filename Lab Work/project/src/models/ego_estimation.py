import torch
import torch.nn as nn
import torch.nn.functional as F

class EgoMotionModel(nn.Module):
    def __init__(self):
        super(EgoMotionModel, self).__init__()

        # Encoder (similar to Depth Estimation, but ends with 256 feature map)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Ego-motion Decoder (predicting 6D vector: 3D translation, 3D rotation as Euler angles)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 6, kernel_size=1)  # Predicts 6 values (3 for translation, 3 for rotation)
        )

    def forward(self, x):
        x = self.encoder(x)
        ego_motion = self.decoder(x)
        return ego_motion.view(-1, 6)  # Flatten to (batch_size, 6)
