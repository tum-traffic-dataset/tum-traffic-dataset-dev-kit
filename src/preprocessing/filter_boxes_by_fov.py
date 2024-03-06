import os
import json
from pathlib import Path
import numpy as np

from src.utils.utils import check_corners_within_image, get_cuboid_corners
from src.utils.detection import Detection, save_to_openlabel
from src.utils.perspective import parse_perspective
from src.utils.vis_utils import VisualizationUtils
from scipy.spatial.transform import Rotation as R

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path_boxes",
        type=str,
        help="Input directory path",
        default="a9_dataset/r01_full_split/test_full/labels_point_clouds/s110_lidar_ouster_south_and_north_merged_with_origin_in_south",
    )

    parser.add_argument(
        "--output_folder_path_boxes_filtered",
        type=str,
        help="Output directory path",
        default="a9_dataset/r01_full_split/test_full/labels_point_clouds/s110_lidar_ouster_south_and_north_merged_with_origin_in_south_filtered_for_south2_fov_with_tracking_in_lidar",
    )

    parser.add_argument(
        "--input_file_path_camera_calib",
        type=str,
        help="Perspective file path",
        default="s110_camera_basler_south2_8mm.json",
    )

    parser.add_argument(
        "--coordinate_system_origin",
        type=str,
        help="Coordinate system origin. Possible values: [s110_lidar_ouster_south, s110_lidar_ouster_north, s110_base]",
        default="s110_lidar_ouster_south",
    )

    parser.add_argument("--track_objects", action="store_true", help="Track objects")
    args = parser.parse_args()

    # NOTE: do only track object on a sequence (e.g. test sequence)
    # iterate all files
    if not os.path.exists(args.output_folder_path_boxes_filtered):
        os.makedirs(args.output_folder_path_boxes_filtered)

    perspective = parse_perspective(args.input_file_path_camera_calib)
    utils = VisualizationUtils()

    for file_name in sorted(os.listdir(args.input_folder_path_boxes)):
        label_data = json.load(open(os.path.join(args.input_folder_path_boxes, file_name)))
        valid_labels = {}
        frame_idx = -1
        detections = []
        for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
            frame_idx = frame_id
            for object_id, label in frame_obj["objects"].items():
                cuboid = np.array(label["object_data"]["cuboid"]["val"])
                if "attributes" in label["object_data"]["cuboid"]:
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "sensor_id"
                    )
                    if attribute is not None:
                        sensor_id = attribute["val"]
                    else:
                        sensor_id = ""
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "body_color"
                    )
                    if attribute is not None:
                        color = attribute["val"]
                    else:
                        color = ""
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["num"], "num_points"
                    )
                    if attribute is not None:
                        num_points = int(float(attribute["val"]))
                    else:
                        num_points = -1

                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["num"], "score"
                    )
                    if attribute is not None:
                        score = round(float(attribute["val"]), 2)
                    else:
                        score = -1

                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "occlusion_level"
                    )
                    if attribute is not None:
                        occlusion_level = attribute["val"]
                    else:
                        occlusion_level = "NOT_OCCLUDED"
                else:
                    sensor_id = ""
                    color = ""
                    num_points = -1
                    score = -1
                detections.append(
                    Detection(
                        uuid=object_id,
                        category=label["object_data"]["type"],
                        location=np.array([[cuboid[0]], [cuboid[1]], [cuboid[2]]]),
                        dimensions=(cuboid[7], cuboid[8], cuboid[9]),
                        yaw=R.from_quat(np.array([cuboid[3], cuboid[4], cuboid[5], cuboid[6]])).as_euler(
                            "xyz", degrees=False
                        )[2],
                        score=score,
                        num_lidar_points=num_points,
                        color=color,
                        occlusion_level=occlusion_level,
                        sensor_id=sensor_id,
                    )
                )
        detections_valid = []
        for detection in detections:
            quaternion = R.from_euler("xyz", [0, 0, detection.yaw], degrees=False).as_quat()
            cuboid = [
                detection.location[0][0],
                detection.location[1][0],
                detection.location[2][0],
                quaternion[0],
                quaternion[1],
                quaternion[2],
                quaternion[3],
                detection.dimensions[0],
                detection.dimensions[1],
                detection.dimensions[2],
            ]
            corners = get_cuboid_corners(cuboid)
            if args.coordinate_system_origin == "s110_lidar_ouster_south":
                corner_points_2d = perspective.project_from_lidar_south_to_image(np.array(corners).T)
            elif args.coordinate_system_origin == "s110_lidar_ouster_north":
                corner_points_2d = perspective.project_from_lidar_north_to_image(np.array(corners).T)
            elif args.coordinate_system_origin == "s110_base":
                corner_points_2d = perspective.project_from_base_to_image(np.array(corners).T)
            else:
                raise ValueError("Unknown coordinate system origin: {}".format(args.lidar_origin))

            if check_corners_within_image(corner_points_2d.T):
                detections_valid.append(detection)
        print("processing file: ", file_name)
        save_to_openlabel(detections_valid, file_name, Path(args.output_folder_path_boxes_filtered))
