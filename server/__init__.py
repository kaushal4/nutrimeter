import os
from flask import Flask
from openai import OpenAI
from inference.segmentor import Segmentor
from inference.vlm_classifier import VLMClassifier
from inference.nutrition import NutritionFinder

def create_app():
    app = Flask(__name__)
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
        sam_path = app.config['SAM_CHECKPOINT']
        app.segmentor = Segmentor(model_path=sam_path)
        
        # Load VLM Classifier
        app.classifier = VLMClassifier()
        
        # Load Nutrition Finder
        app.nutrition_finder = NutritionFinder()

        if not app.config.get('DEEPSEEK_API_KEY'):
            app.logger.warning("FATAL: DEEPSEEK_API_KEY not set. Analysis will fail.")
            app.deepseek_client = None
        else:
            app.deepseek_client = OpenAI(
                api_key=app.config['DEEPSEEK_API_KEY'],
                base_url=app.config['DEEPSEEK_BASE_URL']
            )
            app.logger.info("DeepSeek client initialized.")
        
        app.logger.info("--- All models loaded successfully ---")
        
    except Exception as e:
        app.logger.error(f"FATAL: Error loading ML models: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        app.segmentor = None
        app.classifier = None
        app.nutrition_finder = None
        app.deepseek_client = None # <--- 5. SET TO NONE ON FAILURE
    # --------------------------

    with app.app_context():
        from . import api
        app.register_blueprint(api.bp)

    @app.route('/health')
    def health_check():
        # --- 6. ADD CLIENT TO HEALTH CHECK ---
        if (app.segmentor is None or 
            app.classifier is None or 
            app.nutrition_finder is None or
            app.deepseek_client is None):
            return "ERROR: One or more models not loaded", 500
        return "OK", 200

    return app