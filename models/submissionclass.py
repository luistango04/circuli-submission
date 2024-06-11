import cv2
from ultralytics import YOLO
import json
import numpy as np
import open3d as o3d

from scipy.spatial import cKDTree
from joblib import Parallel, delayed

def jsonprepare(screws,transformation_matrix):
    transformed_screws = []
    for idx, x in enumerate(screws):
        # Transform screw coordinates
        screw_xyz = np.array(
            [x.midpointXYZ[0], x.midpointXYZ[1], x.midpointXYZ[2], 1])  # Extend to homogeneous coordinates
        transformed_xyz = np.dot(transformation_matrix, screw_xyz)[
                          :3]  # Apply transformation and remove homogeneous component

        # Transform screw normals
        screw_normal = np.array([x.normals[0], x.normals[1], x.normals[2],
                                 0])  # Extend to homogeneous coordinates, assuming normals are directions
        transformed_normal = np.dot(transformation_matrix, screw_normal)[
                             :3]  # Apply transformation and remove homogeneous component

        # Append transformed screw coordinates and normals for each screw
        transformed_screws.append({
            "midpointXYZ": transformed_xyz.tolist(),
            "normals": transformed_normal.tolist()
        })

        # Construct JSON payload containing transformed screw poses
        json_payload = {

            "screw_poses": transformed_screws
        }

    # Convert JSON payload to string
        json_string = json.dumps(json_payload, indent=4)

    return json_string


def displaypointclouds(od3object):
    filename = 'static/results.ply'
    # Visualize the first half point cloud
    # o3d.visualization.draw_geometries([od3object],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    o3d.io.write_point_cloud(filename, od3object)
    return filename
def estimate_normals_kdtree(points, k=30, n_jobs=6):
    """
    Estimate the normals of a point cloud using PCA and KD-tree.

    Args:
        points (np.ndarray): The point cloud as an Nx3 array.
        k (int): Number of nearest neighbors to use for normal estimation.
        n_jobs (int): Number of parallel jobs to run. -1 means using all processors.

    Returns:
        np.ndarray: Normals for each point in the point cloud.
    """
    tree = cKDTree(points)
    viewpoint = np.array([0.0, 0.0, 0.0])

    def compute_normal(index, skip=2):
        if index % skip == 0:
            neighbors_idx = tree.query(points[index], k=k)[1]
            neighbors = points[neighbors_idx]
            cov_matrix = np.cov(neighbors.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal = eigenvectors[:, 0]
            # Ensure the normal is facing towards the viewpoint
            to_viewpoint = viewpoint - points[index]
            if np.dot(normal, to_viewpoint) < 0:
                normal = -normal
            return normal
        else:
            return None

        return normal

    normals = Parallel(n_jobs=n_jobs)(delayed(compute_normal)(i) for i in range(points.shape[0]) if compute_normal(i) is not None)

    return np.array(normals)


def create_sphere_at_position(pos, radius=1.0, resolution=20):
    """
    Create a red sphere mesh centered at a specified position.
    Returns:
        open3d.geometry.TriangleMesh: Red sphere mesh centered at the specified position.
    """
    # Create a sphere mesh
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)

    # Translate the sphere to the specified position
    sphere_mesh.translate(pos)

    # Paint the sphere red
    sphere_mesh.paint_uniform_color([1, 0, 0])  # Red color

    return sphere_mesh


def annotate_point_cloud(point_cloud, positions, normals, sphere_radius=30, arrow_length=10):
    """
    Annotates a 3D point cloud with spheres representing positions and arrows representing normals.

    Returns:
        open3d.geometry.PointCloud: Annotated point cloud.
    """
    # Initialize an empty point cloud to store the annotations
    annotated_point_cloud = o3d.geometry.PointCloud()
    annotated_point_cloud.points = point_cloud.points  # Copy original points
    annotated_point_cloud.colors = point_cloud.colors  # Copy original points
    # Add spheres and arrows to the annotated point cloud

    for pos, normal in zip(positions, normals):
        # Create sphere at the position
        sphere_at_pos = create_sphere_at_position(pos, radius=sphere_radius)

        # Add points and triangles of the sphere to the annotated point cloud
        annotated_point_cloud.points.extend(np.asarray(sphere_at_pos.vertices))
        annotated_point_cloud.colors.extend(np.asarray(sphere_at_pos.vertex_colors))

    return annotated_point_cloud




class Screw:
    def __init__(self, cropped_image, bounding_box, confidence, outerclass):
        self.cropped_image = cropped_image
        x_mid = (bounding_box[0] + bounding_box[2]) / 2
        y_mid = (bounding_box[1] + bounding_box[3]) / 2
        self.midpointcloud = [x_mid // 2, y_mid // 2]
        self.bounding_box = bounding_box
        self.outerclass = outerclass
        _ , self.cropmask = self.draw_circleofscrew() # finds circle and replaces midpoint
        self.midpointcloud[0] = int(self.midpointcloud[0])
        self.midpointcloud[1] = int(self.midpointcloud[1])

        self.midpointXYZ,self.normals,self.resized_points,self.resized_colors = self.mask_pointcloud()
        self.xyzrpy = self.transformtoworldframe()

        self.confidence = confidence
    # Function to annoate a 3d Point cloud with sphere representing XYZ and an array representing the normal vector
    def transformtoworldframe(self):
        self.xyzrpy = [self.midpointXYZ,self.normals]
    def draw_circleofscrew(self):
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply a median blur to the image
        blurred_img = cv2.medianBlur(gray_img, 5)
        blurred_img = cv2.equalizeHist(blurred_img)
        blurred_img = cv2.GaussianBlur(blurred_img, (11, 11), cv2.BORDER_DEFAULT)

        # Binarize the image
        _, binarized_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)



        # Detect circles in the closed image using HoughCircles
        circles = cv2.HoughCircles(
            binarized_img,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=100,
            param1=40,
            param2=20,
            minRadius=0,
            maxRadius=25
        )
        mask = np.ones_like(self.cropped_image, dtype=np.uint8) * 255  # White mask
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw filled circles in the mask, setting those regions to black
                cv2.circle(mask, (i[0], i[1]), i[2]+25, (0, 0, 0), thickness=-1)
                self.midpointcloud = [(self.bounding_box[0]+i[0])//2, (self.bounding_box[1] + i[1])//2]

        # Apply the mask to the cropped image
        masked_image = cv2.bitwise_and(self.cropped_image, mask)


        # Ensure at least some circles were found
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(self.cropped_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(self.cropped_image, (i[0] , i[1]), i[2]+15, (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(self.cropped_image, (i[0], i[1]), 2, (0, 0, 255), 3)

        return self.midpointcloud,masked_image

    def save_image(self, file_path):
        cv2.imwrite(file_path + ".jpg", self.cropped_image)
        cv2.imwrite(file_path + "-crop.jpg", self.cropmask)

    def get_bounding_box(self):
        return self.bounding_box

    def get_confidence(self):
        return self.confidence

    def mask_pointcloud(self):
        # Unpack bounding box
        x_min, y_min, x_max, y_max = self.bounding_box
        # Scaling down y_min and y_max by 4
        x_min = int(x_min // 2)
        x_max = int(x_max // 2)
        y_min = int(y_min // 2)
        y_max = int(y_max // 2)

        # Convert Open3D point cloud to numpy arrays for points and colors
        points = np.asarray(self.outerclass.ply_cloud.points)
        colors = np.asarray(self.outerclass.ply_cloud.colors)

        # Reshape the points array if necessary
        height = 1024  # Define or pass the height and width
        width = 1224

        # Reshape points array
        points_reshaped = points.reshape((height, width, -1))
        resized_points = points_reshaped[y_min:y_max, x_min:x_max, :].reshape((-1, 3), order='F')


        midpointXYZ = points_reshaped[self.midpointcloud[1],self.midpointcloud[0],:]
        # Estimate normals
        normals = estimate_normals_kdtree(resized_points, k = 15, n_jobs =7)

        # Reshape colors array
        colors_reshaped = colors.reshape((height, width, -1))
        resized_colors = colors_reshaped[y_min:y_max, x_min:x_max, :].reshape((-1, 3), order='F')
        # Calculate the centroid of the points
        centroid = np.mean(points, axis=0)

        # Calculate the average normal
        average_normal = np.mean(normals, axis=0)
        average_normal /= np.linalg.norm(average_normal)  # Normalize the average normal



        return midpointXYZ,average_normal,resized_points,resized_colors
class FileContainer:
    def __init__(self):

        # detect screws
        self.model = YOLO("best.pt")  # load a pretrained model
    def loaddata(self, ply_file, png_file, json_file):
        self.ply_cloud = self._load_ply_cloud(ply_file)

        self.png_image = self._load_png_image(png_file)
        self.json_data = self._load_json_data(json_file)

        annotated_point_cloud = o3d.geometry.PointCloud()
        annotated_point_cloud.points = self.ply_cloud.points  # Copy original points
        annotated_point_cloud.colors = self.ply_cloud.colors  # Copy original points
        self.annotated_point_cloud = annotated_point_cloud

    def transformpointcloud(self):
        transformation_matrix_list  = self.json_data
        transformation_matrix = np.array(transformation_matrix_list)


        self.annotated_point_cloud.transform(transformation_matrix)
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
                singlescrew = Screw(cropped_image, box, conf, self)
                # singlescrew.save_image(f"static/cropped_image_{idx}")

                self.screws.append(singlescrew)


        # Save image with bounding boxes
        cv2.imwrite("static/output_image.jpg", image)

        # Return image with bounding boxes and interest array
        return  "/output_image.jpg", interest

    def annotateallscrews(self):
        # Iterate through all screws inside self.screws and perform the below function


        for idx,x in enumerate(self.screws):
            self.annotated_point_cloud = annotate_point_cloud(self.annotated_point_cloud, [x.midpointXYZ], [x.normals])
            ## add lines of code that will take the json transformation 4 x 4 matrix  and transform the screw xyz and normals

    def _load_ply_cloud(self, ply_file):
        # Load PLY data using Open3D
        if ply_file:
            cloud = o3d.io.read_point_cloud(ply_file)

            #self.visualize_first_half_pointcloud(cloud)
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

