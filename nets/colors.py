import torch
import cv2
import numpy as np

class ColorMaps:
    """Class to handle different color maps for visualization"""
    
    @staticmethod
    def get_black_white_lut(device="cuda"):
        """Create a black-and-white lookup table"""
        lut = torch.arange(0, 256, dtype=torch.float32, device=device).unsqueeze(1)
        lut = (lut >= 128).float() * 255  # Threshold at 128
        return lut.expand(-1, 3)  # Expand to RGB channels
    
    @staticmethod
    def get_jet_lut():
        """Create a lookup table from the Jet colormap"""
        lut = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)
        return torch.from_numpy(lut).float()
    
    @staticmethod
    def get_grayscale_lut(device="cuda"):
        """Create a grayscale lookup table"""
        lut = torch.arange(0, 256, dtype=torch.float32, device=device).unsqueeze(1)
        return lut.expand(-1, 3)  # Expand to RGB channels