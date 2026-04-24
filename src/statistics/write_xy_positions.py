import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../src", ".."))
from src.utils.vis_utils import VisualizationUtils
import numpy as np

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_paths_boxes",
        type=str,
        help="input folder paths to boxes",
        default="",
    )
    arg_parser.add_argument(
        "--output_folder_path_statistic_plots",
        type=str,
        help="output folder path to statistics plots",
        default="",
    )
    arg_parser.add_argument(
        "--dataset_classes",
        type=str,
        help="dataset classes",
        default="tum_traffic")
    args = arg_parser.parse_args()
    # input_folder_path_boxes = args.input_folder_path_boxes
    # parse input folder paths
    input_folder_paths_boxes = args.input_folder_paths_boxes.split(",")
    dataset_classes = args.dataset_classes
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots

    # create output folder if not exists
    if not os.path.exists(output_folder_path_statistic_plots):
        os.makedirs(output_folder_path_statistic_plots)

    utils = VisualizationUtils()

    # TUM Traffic Dataset
    if dataset_classes == "tum_traffic":
        classes = [
            "CAR",
            "TRUCK",
            "TRAILER",
            "VAN",
            "MOTORCYCLE",
            "BUS",
            "PEDESTRIAN",
            "BICYCLE",
            "EMERGENCY_VEHICLE",
            "OTHER",
        ]
    elif dataset_classes == "providentia":
        # Providentia classes
        classes = ["PRE_TRACK",
                   "OTHER",
                   "PEDESTRIAN",
                   "BIKE",
                   "CAR",
                   "TRUCK",
                   "BUS",
                   "CONSTRUCTION_VEHICLE",
                   "DYNAMIC_TRAFFIC_SIGN",
                   "TRAFFICSIGN",
                   "ANIMAL",
                   "OBSTACLE",
                   "CONSTRUCTIONSITEDELIMITER"]
    else:
        raise ValueError("Unknown dataset type")
    classes_valid_set = set()
    valid_ids = set()

    transformation_matrix_s110_lidar_ouster_north_to_south = np.array(
        [
            [9.58895265e-01, -2.83760227e-01, -6.58645965e-05, 1.41849928e00],
            [2.83753514e-01, 9.58874128e-01, -6.65957109e-03, -1.37385689e01],
            [1.95287726e-03, 6.36714187e-03, 9.99977822e-01, 3.87637894e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        dtype=float,
    )

    xy_coords = []
    for input_folder_path_boxes in input_folder_paths_boxes:
        input_file_paths_boxes = sorted(glob.glob(os.path.join(input_folder_path_boxes, "*.json")))
        # raise ValueError("Unknown dataset type")
        for file_path in input_file_paths_boxes:
            file_name = os.path.basename(file_path)
            json_data = json.load(open(file_path))
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                num_labeled_objects = len(frame_obj["objects"].keys())
                for object_track_id, object_json in frame_obj["objects"].items():
                    object_class = object_json["object_data"]["type"]
                    # NOTE: sometimes the dtwin has MOTORCYCLE as class name (coming from YOLOv7), sometimes BIKE (coming from the fusion result)
                    if dataset_classes == "providentia" and object_class == "MOTORCYCLE":
                        object_class = "BIKE"

                    classes_valid_set.add(object_class)
                    valid_ids.add(classes.index(object_class))

                    if "cuboid" in object_json["object_data"]:
                        cuboid = object_json["object_data"]["cuboid"]["val"]
                        location = cuboid[0:3]
                        if "north" in file_name:
                            # transform location from north to south
                            location_homo = np.array([location[0], location[1], location[2], 1])
                            location_transformed = np.matmul(transformation_matrix_s110_lidar_ouster_north_to_south,
                                                              location_homo)
                            xy_coords.append(list(location_transformed[0:2]))
                        else:
                            xy_coords.append(location[0:2])

    # write xy positions to file
    with open(os.path.join(output_folder_path_statistic_plots, "TUMTraf-Intersection_xy_coord.json"), "w") as f:
        json.dump(xy_coords, f)
