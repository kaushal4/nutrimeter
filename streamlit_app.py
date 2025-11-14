import streamlit as st
import numpy as np
import cv2 
from PIL import Image
import os

# Import your custom ML classes
from inference.segmentor import Segmentor
from inference.vlm_classifier import VLMClassifier
from inference.nutrition import NutritionFinder
from inference.estimator_3d import get_dimensions_from_mask

# --- 1. Model Caching ---
# This decorator ensures models are loaded ONLY ONCE.
@st.cache_resource
def load_models():
    """
    Loads and returns all ML models.
    """
    print("--- [Streamlit] Loading ML Models ---")
    
    # Define paths relative to this streamlit_app.py file
    SAM_CHECKPOINT = os.path.join('checkpoints', 'sam_model.pth')
    
    if not os.path.exists(SAM_CHECKPOINT):
        st.error(f"FATAL: SAM model not found at {SAM_CHECKPOINT}. Please download it.")
        return None, None, None
        
    segmentor = Segmentor(model_path=SAM_CHECKPOINT)
    classifier = VLMClassifier()
    nutrition_finder = NutritionFinder()
    
    print("--- [Streamlit] All models loaded successfully ---")
    return segmentor, classifier, nutrition_finder

# --- 2. Pipeline Logic ---
# This function contains all the processing logic from your API
def run_full_pipeline(image_np, segmentor, classifier, nutrition_finder):
    """
    Runs the complete segmentation, classification, and nutrition
    pipeline on a single image.
    """
    
    # 3.1: Run SAM
    sam_masks = segmentor.generate_masks(image_np)
    
    # 3.2: Run VLM Classifier
    matched_objects = classifier.run_classification(image_np, sam_masks, top_n=5)
    
    # 3.3: Group masks by class name
    grouped_masks = {} 
    for obj in matched_objects:
        label = obj['class_name']
        if label not in grouped_masks:
            grouped_masks[label] = []
        grouped_masks[label].append(obj['segmentation'])

    # 3.4: Merge the grouped masks
    labeled_merged_masks = {} 
    for label, masks_list in grouped_masks.items():
        merged_mask = Segmentor.merge_masks(masks_list)
        if merged_mask.any():
            labeled_merged_masks[label] = merged_mask
    
    # 3.5: Create Output Image
    overlay_image = segmentor.draw_labeled_boundaries(image_np, labeled_merged_masks)

    # 3.6: Create Final JSON & Get Nutrition
    final_json_objects = []
    total_meal_nutrition = {}
    total_valid_area = 0.0
    PIXELS_PER_CM = 50.0 # Placeholder
    
    for i, (label, merged_mask) in enumerate(labeled_merged_masks.items()):
        if label == "unknown":
            continue

        merged_mask_uint8 = merged_mask.astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(merged_mask_uint8)
        merged_bbox = [x, y, w, h]
        merged_area = int(np.sum(merged_mask))
        
        dimensions = {"shape_assumption": "unknown", "notes": "no 3d model for this class"}
        
        # Call NutritionFinder
        nutrition_profile = nutrition_finder.get_nutrition_profile(label)
        
        if nutrition_profile:
            total_valid_area += merged_area
            
        final_json_objects.append({
            "id": i + 1,
            "class_name": label,
            "confidence": 0.99, 
            "area_pixels": merged_area,
            "bbox": merged_bbox,
            "dimensions_cm": dimensions,
            "nutrition_per_100g": nutrition_profile 
        })

    # 3.7: Calculate Weighted Average
    if total_valid_area > 0 and final_json_objects:
        weighted_nutrition_per_100g = {}
        
        sample_keys = None
        for obj in final_json_objects:
            if obj['nutrition_per_100g']:
                sample_keys = obj['nutrition_per_100g'].keys()
                break
        
        if sample_keys:
            for key in sample_keys: 
                weighted_sum = 0.0
                for obj in final_json_objects:
                    if obj['nutrition_per_100g']:
                        weight_fraction = obj['area_pixels'] / total_valid_area
                        weighted_sum += obj['nutrition_per_100g'].get(key, 0) * weight_fraction
                weighted_nutrition_per_100g[key] = round(weighted_sum, 2)
            
            weighted_nutrition_per_ounce = {
                key: round(value * 0.2835, 2)
                for key, value in weighted_nutrition_per_100g.items()
            }
            
            total_meal_nutrition = {
                "per_100g": weighted_nutrition_per_100g,
                "per_ounce": weighted_nutrition_per_ounce
            }
        else:
             total_meal_nutrition = {"notes": "No nutrition data found for any detected items."}
    else:
        total_meal_nutrition = {"notes": "No items with valid area or nutrition data found."}

    # 3.8: Build the final response
    json_output = {
        "pixels_per_cm_ratio_est": PIXELS_PER_CM,
        "objects_detected": final_json_objects,
        "total_meal_nutrition_est": total_meal_nutrition,
    }
    
    # Return the overlay image and the final JSON
    return overlay_image, json_output

# --- 3. Streamlit UI ---

st.set_page_config(layout="wide")
st.title("🥗 NutriMeter: AI Nutrition Analysis")

# Load models
segmentor, classifier, nutrition_finder = load_models()

# UI for file upload
uploaded_file = st.file_uploader("Upload an image of your meal...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and segmentor is not None:
    # Read the image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    
    # Run the pipeline
    with st.spinner("Analyzing your meal... (This may take a moment)"):
        overlay_image, json_output = run_full_pipeline(
            image_np, 
            segmentor, 
            classifier, 
            nutrition_finder
        )

    st.header("🔬 Your Analysis Results")
    
    # --- Display Results in Columns ---
    col1, col2 = st.columns(2)

    # Column 1: Analyzed Image
    col1.image(overlay_image, caption="Analyzed Meal", use_container_width=True)

    # Column 2: Total Meal Nutrition
    col2.subheader("Estimated Total Nutrition (per 100g)")
    
    nutrition_100g = json_output.get("total_meal_nutrition_est", {}).get("per_100g")
    
    if nutrition_100g:
        # Display key metrics
        c1, c2 = col2.columns(2)
        c1.metric("🔥 Calories (kcal)", nutrition_100g.get("Energy_kcal", 0))
        c2.metric("🍗 Protein (g)", nutrition_100g.get("Protein_g", 0))
        c1.metric("🥑 Fat (g)", nutrition_100g.get("Total lipid (fat)_g", 0))
        c2.metric("🍞 Carbs (g)", nutrition_100g.get("Carbohydrate, by difference_g", 0))
        
        # Display a note about the estimation
        col2.info("Note: This is a weighted average based on the *relative size* of the matched items in the image.")
        
    else:
        col2.warning("Could not calculate total nutrition. No items were matched.")

    # --- Full Width Breakdown ---
    st.subheader("Detected Item Breakdown")
    
    for obj in json_output.get("objects_detected", []):
        class_name = obj['class_name'].title()
        if obj['nutrition_per_100g'] is not None:
            st.success(f"✅ **{class_name}** - Matched in nutrition database.")
        else:
            st.warning(f"⚠️ **{class_name}** - No match found in nutrition database.")

    # --- Debug Expander ---
    with st.expander("Show Full JSON Response"):
        st.json(json_output)
        
elif uploaded_file is None:
    st.info("Please upload an image to get started.")