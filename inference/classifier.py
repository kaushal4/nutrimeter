import torch
from ultralytics import YOLO

class Classifier:
    """
    A class to load a YOLOv8 model and run object detection.
    """
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLO model.
        
        Args:
            model_path (str): The name of the pre-trained YOLO model
                              (e.g., 'yolov8n.pt' for nano, 'yolov8s.pt' for small).
                              The model will be downloaded automatically.
        """
        # Set up the device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pre-trained YOLOv8 model
        print(f"Loading YOLO model ({model_path}) to {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print("YOLO model loaded successfully.")

    def detect_objects(self, image_path: str) -> list:
        """
        Runs object detection on the image.
        
        Args:
            image_path (str): The file path to the image.
            
        Returns:
            list: A list of dictionaries, one for each detected object.
        """
        # Run inference on the source
        # We set verbose=False to quiet the large console output
        results = self.model(image_path, verbose=False)
        
        # We're interested in the results for the first (and only) image
        result = results[0]
        
        # Get the class names (e.g., {0: 'person', 1: 'bicycle', ..., 47: 'apple'})
        class_names = result.names
        
        detections = []
        
        # Iterate over each detected bounding box
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = class_names.get(class_id, "unknown")
            confidence = float(box.conf[0])
            
            # Get bounding box coordinates in [x1, y1, x2, y2] format
            bbox_coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox_coords # [x1, y1, x2, y2]
            })
            
        return detections