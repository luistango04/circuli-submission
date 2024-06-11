import open3d as o3d

def display_point_cloud(filename):
    # Read the point cloud from the file
    pcd = o3d.io.read_point_cloud(filename)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Assuming you have already called the function and obtained the filename
filename = 'static/results.ply'

# Display the point cloud
display_point_cloud(filename)