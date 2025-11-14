import os
from flask import Flask
from inference.segmentor import Segmentor
from inference.vlm_classifier import VLMClassifier
from inference.nutrition import NutritionFinder  # <--- 1. IMPORT NUTRITIONFINDER

def create_app():
    app = Flask(__name__)
    # Assuming config.py is in the root, one level up
    app.config.from_pyfile(os.path.join(os.path.dirname(__file__), '..', 'config.py')) 
    
    try:
        os.makedirs(app.config['UPLOAD_DIR'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
    except OSError as e:
        app.logger.error(f"Error creating directories: {e}")

    # --- Load ML Models ---
    app.logger.info("--- Loading ML Models (this may take a moment) ---")
    try:
        # Load SAM Segmentor
        sam_path = app.config['SAM_CHECKPOINT'] # Using SAM_CHECKPOINT from your config
        app.segmentor = Segmentor(model_path=sam_path)
        
        # Load VLM Classifier
        app.classifier = VLMClassifier()
        
        # --- 2. LOAD THE NUTRITION FINDER ---
        app.nutrition_finder = NutritionFinder()
        
        app.logger.info("--- All models loaded successfully ---")
        
    except Exception as e:
        app.logger.error(f"FATAL: Error loading ML models: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        app.segmentor = None
        app.classifier = None
        app.nutrition_finder = None # <--- 3. SET TO NONE ON FAILURE
    # --------------------------

    with app.app_context():
        from . import api
        app.register_blueprint(api.bp)

    @app.route('/health')
    def health_check():
        # --- 4. ADD NUTRITIONFINDER TO HEALTH CHECK ---
        if app.segmentor is None or app.classifier is None or app.nutrition_finder is None:
            return "ERROR: One or more models not loaded", 500
        return "OK", 200

    return app