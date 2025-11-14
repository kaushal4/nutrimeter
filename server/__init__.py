# /3d-vision-api/server/__init__.py

import os
from flask import Flask

def create_app():
    """
    Application factory function to create and configure the Flask app.
    """
    app = Flask(__name__)
    
    # Load configuration from the config.py file
    app.config.from_pyfile(os.path.join('..', 'config.py'))
    
    # Ensure the upload and output directories exist
    try:
        os.makedirs(app.config['UPLOAD_DIR'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
    except OSError as e:
        app.logger.error(f"Error creating directories: {e}")

    # Register the API blueprint
    with app.app_context():
        from . import api
        app.register_blueprint(api.bp)

    @app.route('/health')
    def health_check():
        """A simple health check endpoint."""
        return "OK", 200

    return app