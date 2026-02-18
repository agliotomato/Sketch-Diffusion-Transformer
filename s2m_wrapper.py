import os
import torch
import numpy as np
import cv2
from SketchHairSalon.models.Unet_At_Bg import UnetAtGenerator
import torchvision.transforms.functional as tf

class S2MModel:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = UnetAtGenerator(1, 1, 8, 64, use_dropout=True)
        
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"S2M Model loaded from {checkpoint_path}")
            self.loaded = True
        else:
            print(f"Warning: S2M checkpoint not found at {checkpoint_path}")
            self.loaded = False

    def predict_matte(self, sketch_image):
        """
        Predicts matte from a sketch image (numpy array, grayscale).
        Returns a matte image (numpy array, grayscale, 255).
        """
        if not self.loaded:
            raise RuntimeError("S2M Model is not loaded properly.")

        # Preprocessing (following S2M_test.py)
        # Ensure input is grayscale and correct shape
        if len(sketch_image.shape) == 3:
            sketch_image = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2GRAY)
        
        # Resize or pad if necessary? S2M usually expects 512x512 or similar but fully convolutional
        # Assuming input is already resized to 1024x1024 or whatever SD3.5 uses.
        # However, S2M was likely trained on 512x512. For now, let's keep original re-scaling logic if any.
        # S2M_test.py directly converts to tensor.

        inputs_tensor = tf.to_tensor(sketch_image[:, :, np.newaxis]) * 2.0 - 1.0
        inputs_tensor = inputs_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            result_tensor = self.model(inputs_tensor)
            # Postprocessing
            result = ((result_tensor[0] + 1) / 2 * 255).cpu().numpy().transpose(1, 2, 0).astype("uint8")[..., 0]
            
            # Thresholding for cleaner matte (from S2M_test.py)
            result[result > 250] = 255
            
        return result
