from flask import Flask, request, redirect, url_for, render_template
from models.submissionclass import FileContainer  # Import the FileContainer class

import os
#
# def main():
#     # Define the directory containing the data
#     data_dir = r'D:\OneDrive\Desktop\Documents\Assessment\screw_detection_challenge\trainingdata\MAN_ImgCap_closer_zone_50'
#
#     # Define file names for the PLY, PNG, and JSON files
#     ply_file = os.path.join(data_dir, 'MAN_ImgCap_closer_zone_50.ply')
#     png_file = os.path.join(data_dir, 'MAN_ImgCap_closer_zone_50.png')
#     json_file = os.path.join(data_dir, 'MAN_ImgCap_closer_zone_50.json')
#
#     # Initialize FileContainer object
#     file_container = FileContainer()
#
#     # Load data from the specified files
#     file_container.loaddata(ply_file, png_file, json_file)
#
#     # Detect screws and save the output image with bounding boxes
#     output_image, interest = file_container.detectscrews()
#
#     print(f"Detected screws: {len(file_container.screws)}")
#     for idx, screw in enumerate(file_container.screws):
#         print(f"Screw {idx + 1}:")
#         print(f"  Bounding box: {screw.get_bounding_box()}")
#         print(f"  Confidence: {screw.get_confidence()}")
#
#     # Visualize the first half of the point cloud
#     #file_container.visualize_first_half_pointcloud(file_container.ply_cloud)
#
#
# if __name__ == "__main__":
#     main()
