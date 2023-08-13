import numpy as np

np.set_printoptions(suppress=True)
from argparse import ArgumentParser
import pandas as pd


def read_point_cloud_with_intensity(input_file_path):
    with open(input_file_path, "r") as reader:
        lines = reader.readlines()
    header = lines[:11]
    point_cloud_array = np.array(pd.read_csv(input_file_path, sep=" ", skiprows=11, dtype=float).values)[:, :4]
    return point_cloud_array, header


def write_point_cloud_with_intensity_and_rgb(output_file_path, point_cloud_array, header):
    # update num points
    header[2] = "FIELDS x y z intensity rgb\n"
    header[3] = "SIZE 4 4 4 4 4\n"
    header[4] = "TYPE F F F F F\n"
    header[5] = "COUNT 1 1 1 1 1\n"
    header[6] = "WIDTH " + str(len(point_cloud_array)) + "\n"
    header[7] = "HEIGHT 1" + "\n"
    header[9] = "POINTS " + str(len(point_cloud_array)) + "\n"
    with open(output_file_path, "w") as writer:
        for header_line in header:
            writer.write(header_line)
    df = pd.DataFrame(point_cloud_array)
    df.to_csv(output_file_path, sep=" ", header=False, mode="a", index=False)


def write_point_cloud_with_intensity(output_file_path, point_cloud_array, header=None):
    if header is None:
        header = [
            "# .PCD v0.7 - Point Cloud Data file format\n",
            "VERSION 0.7\n",
            "FIELDS x y z intensity\n",
            "SIZE 4 4 4 4\n",
            "TYPE F F F F\n",
            "COUNT 1 1 1 1\n",
            "WIDTH " + str(len(point_cloud_array)) + "\n" "HEIGHT 1\n",
            "VIEWPOINT 0 0 0 1 0 0 0\n",
            "POINTS " + str(len(point_cloud_array)) + "\n" "DATA ascii\n",
        ]
    else:
        # update num points
        header[2] = "FIELDS x y z intensity\n"
        header[3] = "SIZE 4 4 4 4\n"
        header[4] = "TYPE F F F F\n"
        header[5] = "COUNT 1 1 1 1\n"
        header[6] = "WIDTH " + str(len(point_cloud_array)) + "\n"
        header[7] = "HEIGHT 1" + "\n"
        header[9] = "POINTS " + str(len(point_cloud_array)) + "\n"
    with open(output_file_path, "w") as writer:
        for header_line in header:
            writer.write(header_line)
    df = pd.DataFrame(point_cloud_array)
    df.to_csv(output_file_path, sep=" ", header=False, mode="a", index=False)


def normalize_intensity(point_cloud_array):
    # normalize intensities
    point_cloud_array[:, 3] *= 1 / point_cloud_array[:, 3].max()


def filter_point_cloud(point_cloud, max_distance):
    # remove zero rows
    point_cloud = point_cloud[~np.all(point_cloud[:, :3] == 0.0, axis=1)]

    # remove nans
    point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1), :]

    # remove points above 200 m distance (robosense) and above 120 m distance (ouster)
    distances = np.array(
        [np.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]) for point in point_cloud]
    )
    point_cloud = point_cloud[distances < max_distance, :]
    return point_cloud


def parse_parameters():
    parser = ArgumentParser()
    parser.add_argument(
        "--folder_path_point_cloud_source",
        default="point_clouds_source",
        help="folder path of source point cloud (will be transformed to target point cloud frame)",
    )
    parser.add_argument(
        "--folder_path_point_cloud_target",
        default="point_clouds_target",
        help="folder path of target point cloud (remains static and will not be transformed)",
    )
    parser.add_argument("--initial_voxel_size", type=float, default=2, help="initial voxel size")
    parser.add_argument("--continuous_voxel_size", type=float, default=2, help="continuous voxel size")
    parser.add_argument(
        "--save_registered_point_clouds",
        action="store_true",
        help="Save registered point cloud (By default it is not saved)",
    )
    parser.add_argument(
        "--output_folder_path_registered_point_clouds",
        default="output",
        help="Output folder path to save registered point clouds (default: output)",
    )
    args = parser.parse_args()
    return args
