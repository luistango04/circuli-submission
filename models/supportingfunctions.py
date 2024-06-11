
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



