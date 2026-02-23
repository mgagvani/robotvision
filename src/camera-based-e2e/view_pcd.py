import open3d as o3d
import sys

# Usage: python view_pcd.py filename.pcd
if len(sys.argv) < 2:
    print("Please provide a filename.")
else:
    filename = sys.argv[1]
    print(f"Loading {filename}...")
    pcd = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([pcd])