import cv2
import pyntcloud
from ultralytics import YOLO
import json
import numpy as np

class Screw:
    def __init__(self, cropped_image, bounding_box, confidence):
        self.cropped_image = cropped_image
        self.draw_circleofscrew()
        self.bounding_box = bounding_box
        self.confidence = confidence

    def draw_circleofscrew(self):
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply a median blur to the image
        blurred_img = cv2.medianBlur(gray_img, 5)

        # Binarize the image
        _, binarized_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Perform morphological closing to fill in the circles
        # kernel = np.ones((5, 5), np.uint8)
        # closed_img = cv2.morphologyEx(binarized_img, cv2.MORPH_CLOSE, kernel)

        # Detect circles in the closed image using HoughCircles
        circles = cv2.HoughCircles(
            binarized_img,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=100,
            param1=40,
            param2=20,
            minRadius=0,
            maxRadius=0
        )

        # Ensure at least some circles were found
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(self.cropped_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(self.cropped_image, (i[0], i[1]), 2, (0, 0, 255), 3)

        return self.cropped_image

    def save_image(self, file_path):
        cv2.imwrite(file_path, self.cropped_image)

    def get_bounding_box(self):
        return self.bounding_box

    def get_confidence(self):
        return self.confidence


class FileContainer:
    def __init__(self):

        # detect screws
        self.model = YOLO("best.pt")  # load a pretrained model

    def loaddata(self, ply_file, png_file, json_file):
        self.ply_cloud = self._load_ply_cloud(ply_file)
        self.png_image = self._load_png_image(png_file)
        self.json_data = self._load_json_data(json_file)
    def detectscrews(self):
        # Read the PNG image
        self.screws = []
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
        for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
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
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)

            # Check if the detected object is a screw (assuming class 1 is screw, change as needed)
            if cls == 1:  # Replace with the appropriate class index for screws
                # Crop the screw image from the bounding box
                x1, y1, x2, y2 = map(int, box)
                cropped_image = image[y1:y2, x1:x2]

                # Create a Screw object
                singlescrew = Screw(cropped_image, box, conf)
                singlescrew.save_image(f"static/cropped_image_{idx}.jpg")

                self.screws.append(singlescrew)


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

