import os
import numpy as np
from flask import (
    Blueprint, 
    request, 
    jsonify, 
    current_app, 
    send_from_directory
)
from werkzeug.utils import secure_filename
from inference.estimator_3d import get_dimensions_from_mask
from inference.segmentor import Segmentor # Segmentor is needed for merge_masks
import cv2 # We need this for cv2.boundingRect

bp = Blueprint('api', __name__, url_prefix='/api')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/predict', methods=['POST'])
def predict():
    # 1. --- File Upload (unchanged) ---
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or no file selected"}), 400

    # 2. --- Save File (unchanged) ---
    filename = secure_filename(file.filename)
    upload_path = os.path.join(current_app.config['UPLOAD_DIR'], filename)
    file.save(upload_path)
    
    # 3. --- Run the Full ML Pipeline ---
    try:
        segmentor = current_app.segmentor
        classifier = current_app.classifier
        # --- 3.1: Get the new nutrition_finder ---
        nutrition_finder = current_app.nutrition_finder 
        
        if segmentor is None or classifier is None or nutrition_finder is None:
            return jsonify({"error": "Models not loaded. Check server logs."}), 500

        PIXELS_PER_CM = 50.0 
        
        # 3.2: Run SAM (unchanged)
        image_np = segmentor.load_image(upload_path)
        sam_masks = segmentor.generate_masks(image_np)
        
        # 3.3: Run VLM Classifier (unchanged)
        matched_objects = classifier.run_classification(image_np, sam_masks, top_n=5)
        
        # 3.4: Group masks by class name (unchanged)
        grouped_masks = {} 
        for obj in matched_objects:
            label = obj['class_name']
            if label not in grouped_masks:
                grouped_masks[label] = []
            grouped_masks[label].append(obj['segmentation'])

        # 3.5: Merge the grouped masks (unchanged)
        labeled_merged_masks = {} 
        for label, masks_list in grouped_masks.items():
            merged_mask = Segmentor.merge_masks(masks_list)
            if merged_mask.any():
                labeled_merged_masks[label] = merged_mask
        
        # 3.6: Create Output Image (unchanged)
        overlay_image = segmentor.draw_labeled_boundaries(image_np, labeled_merged_masks)
        
        # 3.7: Save Output Image (unchanged)
        output_filename = f"out_{filename}"
        output_path = os.path.join(current_app.config['OUTPUT_DIR'], output_filename)
        segmentor.save_image(overlay_image, output_path)

        # --- 3.8: NEW STEP: Create Final JSON & Get Nutrition ---
        
        final_json_objects = []
        total_meal_nutrition = {}
        total_valid_area = 0.0  # Total area of items we have nutrition data for
        
        for i, (label, merged_mask) in enumerate(labeled_merged_masks.items()):
            if label == "unknown":
                continue

            # Find bbox [x, y, w, h] of the *merged* mask
            merged_mask_uint8 = merged_mask.astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(merged_mask_uint8)
            merged_bbox = [x, y, w, h]
            merged_area = int(np.sum(merged_mask))
            
            # Get 3D dimensions
            dimensions = None
            if label in ['apple', 'orange', 'plum']:
                dimensions = get_dimensions_from_mask(merged_bbox, PIXELS_PER_CM)
            else:
                dimensions = {"shape_assumption": "unknown", "notes": "no 3d model for this class"}
            
            # --- 3.8.1: Call NutritionFinder ---
            # This returns the nutrition profile per 100g
            nutrition_profile = nutrition_finder.get_nutrition_profile(label)
            
            # --- 3.8.2: Sum the area if nutrition was found ---
            if nutrition_profile:
                total_valid_area += merged_area
                
            final_json_objects.append({
                "id": i + 1,
                "class_name": label,
                "confidence": 0.99, 
                "area_pixels": merged_area,
                "bbox": merged_bbox,
                "dimensions_cm": dimensions,
                "nutrition_per_100g": nutrition_profile # Add per-item nutrition
            })

        # --- 3.9: NEW STEP: Calculate Weighted Average for Total Meal ---
        if total_valid_area > 0 and final_json_objects:
            weighted_nutrition_per_100g = {}
            
            # Get the list of all nutrient keys (e.g., "Energy_kcal", "Protein_g")
            # from the first item that has valid nutrition data
            sample_keys = None
            for obj in final_json_objects:
                if obj['nutrition_per_100g']:
                    sample_keys = obj['nutrition_per_100g'].keys()
                    break
            
            if sample_keys:
                for key in sample_keys: 
                    weighted_sum = 0.0
                    for obj in final_json_objects:
                        # Only include items that have a nutrition profile
                        if obj['nutrition_per_100g']:
                            # Calculate this item's fraction of the total "valid" area
                            weight_fraction = obj['area_pixels'] / total_valid_area
                            # Add its weighted contribution
                            weighted_sum += obj['nutrition_per_100g'].get(key, 0) * weight_fraction
                    
                    # Round to 2 decimal places for cleanliness
                    weighted_nutrition_per_100g[key] = round(weighted_sum, 2)
                
                # --- 3.10: NEW STEP: Convert to "Per Ounce" ---
                # 1 ounce = 28.35g. So, (value / 100g) * 28.35g
                weighted_nutrition_per_ounce = {
                    key: round(value * 0.2835, 2)
                    for key, value in weighted_nutrition_per_100g.items()
                }
                
                total_meal_nutrition = {
                    "notes": f"Estimated nutrition based on a weighted average of {len(weighted_nutrition_per_100g)} items.",
                    "per_100g": weighted_nutrition_per_100g,
                    "per_ounce": weighted_nutrition_per_ounce
                }
            else:
                 total_meal_nutrition = {"notes": "No nutrition data found for any detected items."}
        else:
            total_meal_nutrition = {"notes": "No items with valid area or nutrition data found."}

        # --- 3.11: Build the final response ---
        json_output = {
            "pixels_per_cm_ratio_est": PIXELS_PER_CM,
            "objects_detected": final_json_objects,
            "total_meal_nutrition_est": total_meal_nutrition, # Add the new block
            "output_image_url": f"/api/outputs/{output_filename}"
        }
        
        return jsonify(json_output), 200

    except Exception as e:
        current_app.logger.error(f"Error during prediction: {e}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error during processing"}), 500


@bp.route('/outputs/<string:filename>')
def get_output_file(filename):
    # (This function is unchanged)
    return send_from_directory(
        current_app.config['OUTPUT_DIR'], 
        filename,
        as_attachment=False
    )