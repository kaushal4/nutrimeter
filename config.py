# /3d-vision-api/config.py

import os

# Get the absolute path of the project's root directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- Model Paths ---
# Define the directory where model checkpoints are stored
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
SAM_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'sam_model.pth')
MESH_RCNN_MODEL_PATH = os.path.join(CHECKPOINT_DIR, '3d_model.pth') # Example path for your 3D model

# --- File Storage Paths ---
# Directory for temporarily storing user uploads
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
# Directory for storing generated output images
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# --- Server Settings ---
# You can add other Flask settings here, e.g., DEBUG mode
DEBUG = True