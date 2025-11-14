import os
from flask import Flask
from inference.segmentor import Segmentor
from inference.vlm_classifier import VLMClassifier  # <-- 1. IMPORT NEW

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile(os.path.join('..', 'config.py'))
    
    try:
        os.makedirs(app.config['UPLOAD_DIR'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
    except OSError as e:
        app.logger.error(f"Error creating directories: {e}")

    # --- Load ML Models ---
    try:
        # Load SAM Segmentor (unchanged)
        sam_path = app.config['SAM_MODEL_PATH']
        app.segmentor = Segmentor(model_path=sam_path)
        
        # 2. LOAD THE VLM CLASSIFIER
        # It will download the model on its first run
        app.classifier = VLMClassifier()
        
    except Exception as e:
        app.logger.error(f"FATAL: Error loading ML models: {e}")
        app.segmentor = None
        app.classifier = None
    # --------------------------

    with app.app_context():
        from . import api
        app.register_blueprint(api.bp)

    @app.route('/health')
    def health_check():
        if app.segmentor is None or app.classifier is None:
            return "ERROR: Models not loaded", 500
        return "OK", 200

    return app