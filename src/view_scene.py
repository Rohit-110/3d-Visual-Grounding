import torch
import open3d as o3d
import numpy as np
import warnings

# Optional: suppress Open3D warnings
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# Load the .pth file
file_path = r"C:\Users\SAKETH\Downloads\EaSe\data\referit3d\scan_data\pcd_with_global_alignment\scene0131_00.pth"
data = torch.load(file_path, map_location='cpu', weights_only=False)

# Unpack
points, colors, _, instance_labels = data

# Convert to numpy if needed
if torch.is_tensor(points):
    points = points.numpy()
if torch.is_tensor(colors):
    colors = colors.numpy()

# Normalize colors if needed
if colors.max() > 1.0:
    colors = colors / 255.0

# Create point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
print("[INFO] Visualizing scene0131_00.pth ...")
o3d.visualization.draw_geometries([pcd])  # ✅ Now used after pcd is defined
