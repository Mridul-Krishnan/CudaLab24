import torch
import torch.nn as nn

class EgoMotionModel(nn.Module):
    def __init__(self):
        super(EgoMotionModel, self).__init__()

        # Convolutional layers (input: 6 channels, output: 256 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),  # Input = 6 channels (stacked frames)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer (output: 6-DOF for translation + rotation)
        # The input size to the FC layer will be calculated dynamically based on the output of the encoder
        self.fc = nn.Linear(256 * 8 * 8, 6)  # Placeholder, will be updated dynamically later

    def forward(self, x):
        x = self.encoder(x)

        # Flatten the output of the encoder
        x = x.view(x.size(0), -1)  # Flatten the output for the FC layer

        # Pass through the fully connected layer
        ego_motion = self.fc(x)  # Output: 6-DOF (3 translation, 3 rotation)
        return ego_motion.view(-1, 6)  # Return a 6-element vector (3 for translation, 3 for rotation)

    def update_fc_layer(self, input_shape, device):
        """ Updates the FC layer based on the input shape after convolutional layers """
        # Calculate the flattened size after the convolution layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape).to(device)  # Create a dummy input on the correct device
            conv_output = self.encoder(dummy_input)  # Forward pass through the encoder
            flattened_size = conv_output.numel()  # Get the number of elements after flattening
        
        # Update the FC layer to match the flattened size
        self.fc = nn.Linear(flattened_size, 6).to(device)  # Ensure the FC layer is also on the correct device

