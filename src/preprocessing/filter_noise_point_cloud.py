import open3d as o3d
import numpy as np
import os
import shutil
import argparse

from src.utils.point_cloud_registration_utils import read_point_cloud_with_intensity, write_point_cloud_with_intensity

# Description: Remove noise (outliers) from point clouds and copy non-point cloud files
# Example: python a9-dataset-dev-kit/src/preprocessing/filter_noise_point_cloud.py --input_folder_path_point_clouds <INPUT_FOLDER_PATH_POINT_CLOUDS> \
#                                                                                --output_folder_path_point_clouds <OUTPUT_FOLDER_PATH_POINT_CLOUDS> \
#                                                                                --nb_points 1 \
#                                                                                --radius 0.4

def process_point_cloud(input_path, output_path, nb_points, radius):
    point_cloud_array, header = read_point_cloud_with_intensity(input_path)
    xyz = point_cloud_array[:, :3]
    intensities = point_cloud_array[:, 3]
    max_intensity = np.max(intensities)
    intensities_norm = intensities / max_intensity
    intensities_norm_three_col = np.c_[intensities_norm, intensities_norm, intensities_norm]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(intensities_norm_three_col)

    print("num points before: ", str(len(pcd.points)))
    pcd_filtered, indices_keep = pcd.remove_radius_outlier(nb_points, radius)
    print("num points after: ", str(len(pcd.points)))
    print("removed ", str(len(pcd.points) - len(indices_keep)), " outliers.")

    points_array = np.asarray(pcd_filtered.points)
    intensity_array = np.asarray(pcd.colors)
    # filter intensity array
    intensity_array = intensity_array[indices_keep]
    point_cloud_array = np.c_[points_array, intensity_array[:, 0]]
    write_point_cloud_with_intensity(output_path,point_cloud_array, header)

# Supports traversing multi-layer folders
def process_directory(input_path, output_path, nb_points, radius):
    if not os.path.exists(output_path):
        # os.mkdir(output_path)
        os.makedirs(output_path, exist_ok=True) # can deal with intermediate folders

    for file_name in os.listdir(input_path):
        src = os.path.join(input_path, file_name)
        dst = os.path.join(output_path, file_name)
        if os.path.isdir(src):
            process_directory(src, dst, nb_points, radius)
        elif src.endswith('.pcd'):
            process_point_cloud(src, dst, nb_points, radius)
        else:
            shutil.copy(src, dst)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--input_folder_path_point_clouds", default="point_clouds", type=str, help="Input folder path of point clouds"
    )
    argparser.add_argument(
        "--output_folder_path_point_clouds", default="output", type=str, help="Output folder path of filtered point clouds"
    )
    argparser.add_argument(
        "--nb_points", default=1, type=int, help="Minimum number of points in sphere"
    )
    argparser.add_argument(
        "--radius", default=0.4, type=float, help="Radius of sphere to check point density"
    )

    args = argparser.parse_args()
    process_directory(args.input_folder_path_point_clouds, args.output_folder_path_point_clouds, args.nb_points, args.radius)


