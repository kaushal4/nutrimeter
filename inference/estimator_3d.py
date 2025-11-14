import numpy as np
import math

def get_dimensions_from_mask(mask_bbox: list, pixels_per_cm: float) -> dict:
    """
    Estimates 3D dimensions based on a 2D bounding box and a known scale.
    
    This function makes a strong assumption that the object is roughly
    spherical and its 2D bounding box represents its main diameter.
    
    Args:
        mask_bbox (list): The bounding box of the mask [x, y, width, height].
        pixels_per_cm (float): The conversion factor for this image.
        
    Returns:
        dict: A dictionary with estimated 3D dimensions and volume.
    """
    
    # 1. Get pixel dimensions from the bounding box
    pixel_width = mask_bbox[2]
    pixel_height = mask_bbox[3]
    
    # 2. Assume it's a sphere, so diameter is the average of width and height
    pixel_diameter = (pixel_width + pixel_height) / 2.0
    
    # 3. Convert pixel diameter to real-world centimeters
    diameter_cm = pixel_diameter / pixels_per_cm
    radius_cm = diameter_cm / 2.0
    
    # 4. Calculate volume of a sphere: V = (4/3) * pi * r^3
    volume_cc = (4/3) * math.pi * (radius_cm ** 3)
    
    # 5. Return the results, rounded for clarity
    return {
        "shape_assumption": "sphere",
        "length_cm": round(diameter_cm, 2),
        "width_cm": round(diameter_cm, 2),
        "height_cm": round(diameter_cm, 2),
        "volume_cc": round(volume_cc, 2)
    }