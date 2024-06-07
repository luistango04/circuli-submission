import cv2
import pyntcloud
from ultralytics import YOLO
import json

class FileContainer:
    def __init__(self):

        # detect screws
        self.model = YOLO("yolov8n.pt")  # load a pretrained model
    def loaddata(self, ply_file, png_file, json_file):
        self.ply_cloud = self._load_ply_cloud(ply_file)
        self.png_image = self._load_png_image(png_file)
        self.json_data = self._load_json_data(json_file)
    def detectscrews(self):
        # Read the PNG image
        image = self.png_image

        # Perform inference
        results = self.model(image)

        # Initialize an array to store midpoints of bounding boxes
        interest = []

        # Extract bounding boxes, classes, names, and confidences
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        confidences = results[0].boxes.conf.tolist()

        # Iterate through the results
        for box, cls, conf in zip(boxes, classes, confidences):
            # Calculate midpoint of the bounding box
            x_mid = (box[0] + box[2]) / 2
            y_mid = (box[1] + box[3]) / 2

            # Store the bounding box, class, confidence, and midpoint
            interest.append({
                'box': box,
                'class': cls,
                'confidence': conf,
                'midpoint': (x_mid, y_mid)
            })

            # Draw bounding box
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # Save image with bounding boxes
        cv2.imwrite("static/output_image.jpg", image)

        # Return image with bounding boxes and interest array
        return  "/output_image.jpg", interest



    def _load_ply_cloud(self, ply_file):
        # Load PLY data as PCL cloud
        if ply_file:

            cloud = pyntcloud.PyntCloud.from_file(ply_file)

            return cloud
        else:
            return None

    def _load_png_image(self, png_file):
        # Load PNG data as CV image
        if png_file:
            image = cv2.imread(png_file)
            return image
        else:
            return None

    def _load_json_data(self, json_file):
        # Load JSON data as dictionary
        if json_file:
            with open(json_file, 'r') as file:
                data = json.load(file)
            return data
        else:
            return None

# Example usage:
# file_container = FileContainer(ply_file_path, png_file_path, json_file_path)
# Access the stored data
# ply_cloud_data = file_container.ply_cloud
# png_image_data = file_container.png_image
# json_data = file_container.json_data

