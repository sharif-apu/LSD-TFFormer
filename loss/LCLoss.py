import torch
import torch.nn as nn
from skimage import io, color
import numpy as np
class LuminanceChrominanceLoss(nn.Module):
    def __init__(self, luminance_weight=1.0, chrominance_weight=1.0):
        super(LuminanceChrominanceLoss, self).__init__()
        self.luminance_weight = luminance_weight
        self.chrominance_weight = chrominance_weight

    def forward(self, x, y):
        # Calculate luminance of the images
        luminance_x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        luminance_y = 0.299 * y[:, 0, :, :] + 0.587 * y[:, 1, :, :] + 0.114 * y[:, 2, :, :]

        # Calculate chrominance by subtracting luminance from each RGB channel
        chrominance_x = x[:, 0:3, :, :] - luminance_x.unsqueeze(1)
        chrominance_y = y[:, 0:3, :, :] - luminance_y.unsqueeze(1)

        # Calculate Mean Squared Error (MSE) for luminance and chrominance
        luminance_loss = nn.functional.mse_loss(luminance_x, luminance_y)
        chrominance_loss = nn.functional.mse_loss(chrominance_x, chrominance_y)

        # Combine luminance and chrominance losses with weights
        total_loss = self.luminance_weight * luminance_loss + self.chrominance_weight * chrominance_loss

        return total_loss
