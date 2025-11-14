import torch
import numpy as np
import cv2  # OpenCV for drawing
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import random # For generating random colors

class Segmentor:
    """
    A class to load the SAM model and run segmentation.
    """
    def __init__(self, model_path, model_type="vit_h"):
        """
        Initializes the Segment Anything Model (SAM).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAM model checkpoint not found at: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading SAM model ({model_type}) from {model_path} to {self.device}...")
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.device)
        
        # --- Use the tuned parameters to favor whole objects ---
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            stability_score_thresh=0.92,
            pred_iou_thresh=0.86,
            box_nms_thresh=0.4 # This is key for suppressing duplicate/part masks
        )
        print("SAM model loaded successfully (with custom object-level tuning).")

    def load_image(self, image_path: str) -> np.ndarray:
        # (This function is unchanged)
        image = Image.open(image_path).convert("RGB")
        return np.array(image)

    def generate_masks(self, image_np: np.ndarray) -> list:
        # (This function is unchanged)
        print(f"Generating masks for image with shape {image_np.shape}...")
        masks = self.mask_generator.generate(image_np)
        print(f"Found {len(masks)} masks.")
        return masks

    @staticmethod
    def merge_masks(masks: list[np.ndarray]) -> np.ndarray:
        """
        --- NEW FUNCTION ---
        Merges a list of boolean masks into a single boolean mask
        using a logical OR.
        """
        if not masks:
            return np.zeros((0, 0), dtype=bool) # Return empty
            
        # Get shape from the first mask
        shape = masks[0].shape
        combined = np.zeros(shape, dtype=bool)
        
        for m in masks:
            if m.shape == shape:
                combined |= m # Logical OR
        
        return combined

    @staticmethod
    def draw_labeled_boundaries(image_np: np.ndarray, labeled_merged_masks: dict) -> np.ndarray:
        """
        --- NEW FUNCTION ---
        Draws contours and text labels for a dict of merged masks.
        'labeled_merged_masks' format: {'apple': merged_mask_1, 'pizza': merged_mask_2}
        """
        # Create a copy to draw on
        overlay = image_np.copy()
        
        # Convert to BGR for OpenCV (OpenCV uses BGR, PIL/numpy use RGB)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        for label, merged_mask in labeled_merged_masks.items():
            if label == "unknown": # Don't draw "unknown" objects
                continue
            
            # Generate a random color for this class
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Convert boolean mask to 8-bit image for OpenCV
            mask_uint8 = merged_mask.astype(np.uint8) * 255
            
            # Find contours (outlines) of the merged mask
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue

            # Draw all found contours for this merged object
            cv2.drawContours(overlay, contours, -1, color, 3) # 3px thick line
            
            # --- Add the text label ---
            # Find the top-left corner (x, y) of the bounding box
            # of the largest contour to place the text
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Set font properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_pos = (x, y - 10) # 10px above the top-left corner
            
            # Draw the text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(overlay, (x, y - text_height - 15), (x + text_width, y - 5), color, -1) # Solid fill
            
            # Draw the text
            cv2.putText(overlay, label, text_pos, font, font_scale, (255, 255, 255), font_thickness) # White text

        # Convert back to RGB for PIL/saving
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return overlay

    @staticmethod
    def save_image(image_np: np.ndarray, output_path: str):
        # (This function is unchanged)
        Image.fromarray(image_np).save(output_path)
        print(f"Saved image to {output_path}")