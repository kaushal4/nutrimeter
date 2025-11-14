import torch
import numpy as np
import cv2  # OpenCV for image processing
from PIL import Image  # Pillow for image loading
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

class Segmentor:
    """
    A class to load the SAM model and run segmentation.
    """
    def __init__(self, model_path, model_type="vit_h"):
        """
        Initializes the Segment Anything Model (SAM).
        This loads the (very large) model into memory.
        
        Args:
            model_path (str): Path to the SAM checkpoint file.
            model_type (str): The type of SAM model (e.g., "vit_h").
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAM model checkpoint not found at: {model_path}")

        # Set up the device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        print(f"Loading SAM model ({model_type}) from {model_path} to {self.device}...")
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.device)
        
        # Initialize the automatic mask generator
        # This is the "zero-shot" part that finds all objects
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        print("SAM model loaded successfully.")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Loads an image from a file path and converts it to RGB.
        SAM requires the image in RGB format as a NumPy array.
        
        Args:
            image_path (str): The file path to the image.
            
        Returns:
            np.ndarray: The image as a NumPy array [H, W, 3].
        """
        image = Image.open(image_path).convert("RGB")
        return np.array(image)

    def generate_masks(self, image_np: np.ndarray) -> list:
        """
        Generates segmentation masks for all objects in the image.
        
        Args:
            image_np (np.ndarray): The input image as a NumPy array.
            
        Returns:
            list: A list of dictionaries, where each dict describes one mask.
                  (See SAM docs for 'masks' format)
        """
        print(f"Generating masks for image with shape {image_np.shape}...")
        masks = self.mask_generator.generate(image_np)
        print(f"Found {len(masks)} masks.")
        return masks

    @staticmethod
    def create_overlay_image(image_np: np.ndarray, masks: list) -> np.ndarray:
        """
        Creates a visualization by drawing all masks over the original image.
        Each mask will have a different random color.
        
        Args:
            image_np (np.ndarray): The original image.
            masks (list): The list of mask dictionaries from generate_masks().
            
        Returns:
            np.ndarray: A new image (NumPy array) with masks overlaid.
        """
        # Make a copy of the image to draw on
        overlay = image_np.copy()
        
        if not masks:
            return overlay
            
        # Sort masks by area (largest first) so smaller masks are drawn on top
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        for ann in sorted_masks:
            m = ann['segmentation'] # This is a 2D boolean array
            
            # Generate a random color for this mask
            # We create a 1x3 array and broadcast it to the mask's shape
            color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
            
            # Create a "color mask" by applying the color to the pixels
            # within the segmentation
            mask_color = np.zeros_like(overlay, dtype=np.uint8)
            mask_color[m] = color
            
            # Blend the mask color with the overlay image
            # We use addWeighted for a nice transparency effect
            overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.4, 0)

        return overlay

    @staticmethod
    def save_image(image_np: np.ndarray, output_path: str):
        """
        Saves a NumPy array image to a file.
        
        Args:
            image_np (np.ndarray): The image to save.
            output_path (str): The destination file path.
        """
        # Convert back to PIL Image to save
        Image.fromarray(image_np).save(output_path)
        print(f"Saved image to {output_path}")