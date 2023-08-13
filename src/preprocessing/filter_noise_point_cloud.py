import open3d as o3d
import numpy as np
import os

from pypcd import pypcd
import argparse

from src.utils.point_cloud_registration_utils import read_point_cloud_with_intensity, write_point_cloud_with_intensity

# Description: Remove noise (outliers) from point clouds
# Example: python a9-dataset-dev-kit/src/preprocessing/filter_noise_point_cloud.py --input_folder_path_point_clouds <INPUT_FOLDER_PATH_POINT_CLOUDS> \
#                                                                                  --output_folder_path_point_clouds <OUTPUT_FOLDER_PATH_POINT_CLOUDS> \
#                                                                                  --nb_points 1 \
#                                                                                  --radius 0.4

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--input_folder_path_point_clouds", default="point_clouds", type=str, help="Input folder path of point clouds"
    )
    argparser.add_argument(
        "--output_folder_path_point_clouds",
        default="output",
        type=str,
        help="Output folder path of filtered point clouds",
    )
    argparser.add_argument(
        "--nb_points",
        default="1",
        type=int,
        help="Pick the minimum amount of points that the sphere should contain. Higher nb_points removes more points.",
    )
    argparser.add_argument(
        "--radius",
        default="0.4",
        type=float,
        help="Defines the radius of the sphere that will be used for counting the neighbors.",
    )

    args = argparser.parse_args()
    input_folder_path_point_cloud = args.input_folder_path_point_clouds
    output_folder_path_point_cloud = args.output_folder_path_point_clouds
    nb_points = args.nb_points
    radius = args.radius

    if not os.path.exists(output_folder_path_point_cloud):
        os.mkdir(output_folder_path_point_cloud)

    for file_name in sorted(os.listdir(input_folder_path_point_cloud)):
        point_cloud_array, header = read_point_cloud_with_intensity(
            os.path.join(args.input_folder_path_point_clouds, file_name))
        xyz = point_cloud_array[:, :3]
        intensities = point_cloud_array[:, 3]
        max_intensity = np.max(intensities)
        intensities_norm = np.array(intensities / max_intensity)
        intensities_norm_two_col = np.c_[intensities_norm, intensities_norm]
        intensities_norm_three_col = np.c_[intensities_norm_two_col, intensities_norm]

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
        write_point_cloud_with_intensity(os.path.join(args.output_folder_path_point_clouds, file_name),
                                         point_cloud_array, header)
