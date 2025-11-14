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

bp = Blueprint('api', __name__, url_prefix='/api')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- The calculate_iou() function is GONE! We don't need it. ---

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
        classifier = current_app.classifier # This is now the VLMClassifier
        
        if segmentor is None or classifier is None:
            return jsonify({"error": "Models not loaded. Check server logs."}), 500

        # --- FAKE PIXEL-TO-CM RATIO (unchanged) ---
        PIXELS_PER_CM = 50.0 
        
        # 3.1: Run SAM (unchanged)
        image_np = segmentor.load_image(upload_path)
        sam_masks = segmentor.generate_masks(image_np)
        
        # 3.2: Run VLM Classifier
        # This one line REPLACES the entire YOLO + IoU matching logic
        matched_objects = classifier.run_classification(image_np, sam_masks, top_n=5)
        
        # 3.3: Create Output Image (unchanged)
        # We only pass the *matched* masks to the overlay function
        drawable_masks = []
        for obj in matched_objects:
             drawable_masks.append({
                 'segmentation': obj['segmentation'],
                 'area': obj['area_pixels']
             })
        overlay_image = segmentor.create_overlay_image(image_np, drawable_masks)
        
        # 3.4: Save Output Image (unchanged)
        output_filename = f"out_{filename}"
        output_path = os.path.join(current_app.config['OUTPUT_DIR'], output_filename)
        segmentor.save_image(overlay_image, output_path)

        # 3.5: Create Final JSON (unchanged)
        final_json_objects = []
        for i, obj in enumerate(matched_objects):
            
            # --- 3.6: ESTIMATE 3D DIMS (unchanged) ---
            dimensions = None
            # We can now apply rules to *any* food item
            if obj['class_name'] in ['apple', 'orange', 'plum']:
                dimensions = get_dimensions_from_mask(obj['bbox'], PIXELS_PER_CM)
            else:
                dimensions = {"shape_assumption": "unknown", "notes": "no 3d model for this class"}
            # ----------------------------------------
            
            final_json_objects.append({
                "id": i + 1,
                "class_name": obj['class_name'],
                "confidence": round(obj['confidence'], 4),
                "area_pixels": obj['area_pixels'],
                "bbox": obj['bbox'],
                "dimensions_cm": dimensions
            })

        json_output = {
            "pixels_per_cm_ratio_est": PIXELS_PER_CM,
            "objects_detected": final_json_objects,
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
    return send_from_directory(
        current_app.config['OUTPUT_DIR'], 
        filename,
        as_attachment=False
    )