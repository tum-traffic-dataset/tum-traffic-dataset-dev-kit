import os
import json
from decimal import Decimal
import argparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

import numpy as np
from scipy.spatial.transform import Rotation
from src.utils.vis_utils import VisualizationUtils


# Example usage:
# python merge_boxes_late_fusion.py --input_folder_path_target_boxes <INPUT_FOLDER_PATH_target_BOXES>
#                              --input_folder_path_source_boxes <INPUT_FOLDER_PATH_source_BOXES>
#                              --output_folder_path_fused_boxes <OUTPUT_FOLDER_PATH_FUSED_BOXES>

# Note:
# The result will be stored in specified output folder.
#


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        # If passed in object is instance of Decimal
        # convert it to a string
        if isinstance(obj, Decimal):
            return str(obj)
        # Otherwise use the default behavior
        return json.JSONEncoder.default(self, obj)


def parse_parameters():
    parser = argparse.ArgumentParser(description="Merge 3D boxes (late fusion)")
    parser.add_argument(
        "--input_folder_path_target_boxes",
        type=str,
        default="",
        help="input folder path to target boxes",
    )

    parser.add_argument(
        "--input_folder_path_source_boxes",
        type=str,
        default="",
        help="input folder path to source boxes",
    )

    parser.add_argument(
        "--output_folder_path_fused_boxes",
        type=str,
        default="",
        help="output folder path to fused boxes.",
    )

    parser.add_argument(
        "--same_origin",
        action="store_true",
        help="Specify this flag if the target data is in the same origin frame (coordinate system) as source data. No transformation will be applied.",
    )

    args = parser.parse_args()
    return args


def store_boxes_in_open_label(json_data, output_folder_path, output_file_name):
    with open(os.path.join(output_folder_path, output_file_name), "w", encoding="utf-8") as json_writer:
        json_string = json.dumps(json_data, ensure_ascii=True, indent=4, cls=DecimalEncoder)
        json_writer.write(json_string)


def get_closest_3d_object_by_distance(objects_3d_target, object_3d_source):
    position_3d_source = np.array(object_3d_source["object_data"]["cuboid"]["val"])[:3]
    positions_3d = []
    distances = []

    for object_3d_target in objects_3d_target:
        position_3d_target = np.array(object_3d_target["object_data"]["cuboid"]["val"])[:3]
        distance = np.sqrt(
            (position_3d_source[0] - position_3d_target[0]) ** 2
            + (position_3d_source[1] - position_3d_target[1]) ** 2
        )
        positions_3d.append(position_3d_target)
        distances.append(distance)
    if len(positions_3d) > 0:
        idx_closest = np.argmin(np.array(distances))
        return (
            objects_3d_target[idx_closest],
            positions_3d[idx_closest],
            distances[idx_closest],
        )
    else:
        return None, None, -1


def center_to_center_dist(position_3d_target, position_3d_source):
    distance = np.sqrt(
        (position_3d_source[0] - position_3d_target[0]) ** 2 + (position_3d_source[1] - position_3d_target[1]) ** 2
    )
    return distance


def assignment(objects_target, objects_source, distance_threshold):
    iou_dst = np.zeros((len(objects_target), len(objects_source)))
    for id_s, object_target in enumerate(objects_target):
        position_3d_target = np.array(object_target["object_data"]["cuboid"]["val"])[:3]
        for id_n, object_source in enumerate(objects_source):
            position_3d_source = np.array(object_source["object_data"]["cuboid"]["val"])[:3]
            if not same_origin:
                position_3d_source = np.matmul(
                    transformation_matrix,
                    np.array(
                        [
                            position_3d_source[0],
                            position_3d_source[1],
                            position_3d_source[2],
                            1,
                        ]
                    ),
                )
            distance = center_to_center_dist(position_3d_target[:2], position_3d_source[:2])
            if distance > distance_threshold:
                distance = 999999
            iou_dst[id_s, id_n] = distance

    matched_idx_target, matched_index_source = linear_sum_assignment(iou_dst)
    matched_idx = np.column_stack((matched_idx_target, matched_index_source))

    unmatched_target, unmatched_source = [], []
    for id_s, object_target in enumerate(objects_target):
        if id_s not in matched_idx[:, 0]:
            unmatched_target.append(id_s)

    for id_n, object_source in enumerate(objects_source):
        if id_n not in matched_idx[:, 1]:
            unmatched_source.append(id_n)

    matches = []
    for idx in matched_idx:
        if iou_dst[idx[0], idx[1]] > distance_threshold:
            unmatched_target.append(idx[0])
            unmatched_source.append(idx[1])
        else:
            matches.append(idx.reshape(1, 2))

    return unmatched_target, unmatched_source, matches


if __name__ == "__main__":
    occlusion_name_to_level_mapping = {
        "NOT_OCCLUDED": 0,
        "PARTIALLY_OCCLUDED": 1,
        "MOSTLY_OCCLUDED": 2,
        "": 3,
    }

    occlusion_level_to_name_mapping = {0: "NOT_OCCLUDED", 1: "PARTIALLY_OCCLUDED", 2: "MOSTLY_OCCLUDED", 3: ""}
    args = parse_parameters()
    input_folder_path_target_boxes = args.input_folder_path_target_boxes
    input_folder_path_source_boxes = args.input_folder_path_source_boxes
    output_folder_path_fused_boxes = args.output_folder_path_fused_boxes
    same_origin = bool(args.same_origin)

    if not os.path.exists(output_folder_path_fused_boxes):
        os.mkdir(output_folder_path_fused_boxes)

    input_files_target = sorted(os.listdir(input_folder_path_target_boxes))
    input_files_source = sorted(os.listdir(input_folder_path_source_boxes))

    if (
            "s110_lidar_ouster_south" in input_folder_path_target_boxes
            and "s110_lidar_ouster_north" in input_folder_path_source_boxes
    ):
        # transformation matrix from lidar north to lidar south
        # TODO: use calib files to load transformation matrix
        transformation_matrix = np.linalg.inv(np.array(
            [
                [9.58895265e-01, -2.83760227e-01, -6.58645965e-05, 1.41849928e00],
                [2.83753514e-01, 9.58874128e-01, -6.65957109e-03, -1.37385689e01],
                [1.95287726e-03, 6.36714187e-03, 9.99977822e-01, 3.87637894e-01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            dtype=float,
        ))
    elif (
            "s110_lidar_ouster_north" in input_folder_path_target_boxes
            and "s110_lidar_ouster_south" in input_folder_path_source_boxes
    ):
        transformation_matrix = np.array(
            [
                [9.58895265e-01, -2.83760227e-01, -6.58645965e-05, 1.41849928e00],
                [2.83753514e-01, 9.58874128e-01, -6.65957109e-03, -1.37385689e01],
                [1.95287726e-03, 6.36714187e-03, 9.99977822e-01, 3.87637894e-01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            dtype=float,
        )

    frame_id = 0
    num_matches = 0
    num_objects_target = 0
    num_objects_source = 0
    frame_id = 0
    distance_threshold = 3.0
    for boxes_file_name_target, boxes_file_name_source in zip(input_files_target, input_files_source):
        north_objects = []
        json_label_target = json.load(
            open(
                os.path.join(input_folder_path_target_boxes, boxes_file_name_target),
            )
        )
        json_label_source = json.load(
            open(
                os.path.join(input_folder_path_source_boxes, boxes_file_name_source),
            )
        )

        # get all 3d objects from south lidar
        objects_3d_target_list = []
        for frame_idx, frame_obj in json_label_target["openlabel"]["frames"].items():
            for obj_id, objects_3d_target in frame_obj["objects"].items():
                objects_3d_target_list.append(objects_3d_target)

        # get all 3d objects from north lidar
        objects_3d_source_list = []
        objects_3d_source_ids_hex = []
        for frame_idx, frame_obj in json_label_source["openlabel"]["frames"].items():
            for obj_id, objects_3d_source in frame_obj["objects"].items():
                objects_3d_source_list.append(objects_3d_source)
                objects_3d_source_ids_hex.append(obj_id)

        unmatched_target_ids, unmatched_source_ids, matches_ids = assignment(
            objects_3d_target_list,
            objects_3d_source_list,
            distance_threshold=distance_threshold,
        )

        if len(objects_3d_source_list) > len(objects_3d_target_list):
            print("frame id:", frame_id)

        for matches in matches_ids:
            match_id_target = matches[0, 0]
            match_id_source = matches[0, 1]

            object_3d_target = objects_3d_target_list[match_id_target]
            object_3d_source = objects_3d_source_list[match_id_source]
            num_matches += 1
            # merge lidar north with lidar south 3d position
            # get 3d position from south and north
            position_3d_target = np.array(object_3d_target["object_data"]["cuboid"]["val"])[:3]
            position_3d_source = np.array(object_3d_source["object_data"]["cuboid"]["val"])[:3]

            if not same_origin:
                position_3d_source = np.matmul(
                    transformation_matrix,
                    np.array(
                        [
                            position_3d_source[0],
                            position_3d_source[1],
                            position_3d_source[2],
                            1,
                        ]
                    ),
                )

            position_3d_merged = (
                np.mean([position_3d_target[0], position_3d_source[0]]),
                np.mean([position_3d_target[1], position_3d_source[1]]),
                np.mean([position_3d_target[2], position_3d_source[2]]),
            )

            # get quaternion from target
            quaternion_target = np.array(object_3d_target["object_data"]["cuboid"]["val"])[3:7]
            roll_target, pitch_target, yaw_target = Rotation.from_quat(quaternion_target).as_euler(
                "xyz", degrees=True
            )
            # get quaternion from source
            quaternion_source = np.array(object_3d_source["object_data"]["cuboid"]["val"])[3:7]
            roll_source, pitch_source, yaw_source = Rotation.from_quat(quaternion_source).as_euler(
                "xyz", degrees=True
            )

            if not same_origin:
                delta_yaw = 177.90826842 - 163.58774077
                # TODO: check whether this needs to be subtracted when transforming from south to north
                yaw_source += delta_yaw

            # average rotation
            roll_avg = (roll_target + roll_source) / 2.0
            pitch_avg = (pitch_target + pitch_source) / 2.0
            yaw_avg = (yaw_target + yaw_source) / 2.0

            # convert to quaternion
            quaternion_average = Rotation.from_euler("xyz", [roll_avg, pitch_avg, yaw_avg], degrees=True).as_quat()

            # average dimensions
            # get dimensions from south
            dimensions_3d_target = np.array(object_3d_target["object_data"]["cuboid"]["val"])[-3:]
            # get dimensions from north
            dimensions_3d_source = np.array(object_3d_source["object_data"]["cuboid"]["val"])[-3:]

            dimensions_3d_merged = (
                np.mean([dimensions_3d_target[0], dimensions_3d_source[0]]),
                np.mean([dimensions_3d_target[1], dimensions_3d_source[1]]),
                np.mean([dimensions_3d_target[2], dimensions_3d_source[2]]),
            )

            # average number of points
            if "attributes" in object_3d_source["object_data"]["cuboid"]:
                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_source["object_data"]["cuboid"]["attributes"]["text"],
                    "body_color",
                )
                if attribute is not None:
                    color_source = attribute["val"]
                else:
                    color_source = ""

                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_source["object_data"]["cuboid"]["attributes"]["text"],
                    "occlusion_level",
                )
                if attribute is not None:
                    occlusion_level_source = attribute["val"]
                else:
                    occlusion_level_source = ""

                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_source["object_data"]["cuboid"]["attributes"]["num"],
                    "num_points",
                )
                if attribute is not None:
                    num_points_source = int(attribute["val"])
                else:
                    num_points_source = -1

                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_source["object_data"]["cuboid"]["attributes"]["num"],
                    "score",
                )
                if attribute is not None:
                    score_source = round(float(attribute["val"]), 2)
                else:
                    score_source = -1

                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_source["object_data"]["cuboid"]["attributes"]["num"],
                    "number_of_trailers",
                )
                if attribute is not None:
                    number_of_trailers_source = int(attribute["val"])
                else:
                    number_of_trailers_source = 0
            else:
                color_source = ""
                occlusion_level_source = ""
                num_points_source = 0
                score_source = 0
                number_of_trailers_source = 0

            if "attributes" in object_3d_target["object_data"]["cuboid"]:
                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_target["object_data"]["cuboid"]["attributes"]["text"],
                    "body_color",
                )
                if attribute is not None:
                    color_target = attribute["val"]
                else:
                    color_target = ""

                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_target["object_data"]["cuboid"]["attributes"]["text"],
                    "occlusion_level",
                )
                if attribute is not None:
                    occlusion_level_target = attribute["val"]
                else:
                    occlusion_level_target = ""

                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_target["object_data"]["cuboid"]["attributes"]["num"],
                    "num_points",
                )
                if attribute is not None:
                    num_points_target = int(attribute["val"])
                else:
                    num_points_target = 0

                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_target["object_data"]["cuboid"]["attributes"]["num"],
                    "score",
                )
                if attribute is not None:
                    score_target = round(float(attribute["val"]), 2)
                else:
                    score_target = -1
                attribute = VisualizationUtils.get_attribute_by_name(
                    object_3d_target["object_data"]["cuboid"]["attributes"]["num"],
                    "number_of_trailers",
                )
                if attribute is not None:
                    number_of_trailers_target = int(attribute["val"])
                else:
                    number_of_trailers_target = 0
            else:
                color_target = ""
                occlusion_level_target = ""
                num_points_target = 0
                score_target = 0
                number_of_trailers_target = 0

            num_points_merged = (num_points_target + num_points_source) / 2.0
            score_merged = (score_target + score_source) / 2.0
            if score_target > score_source:
                color_merged = color_target
                number_of_trailers_merged = number_of_trailers_target
            else:
                color_merged = color_source
                number_of_trailers_merged = number_of_trailers_source

            if occlusion_level_target == occlusion_level_source:
                occlusion_level_merged = occlusion_level_target
            else:
                occlusion_level_target = occlusion_name_to_level_mapping[occlusion_level_target]
                occlusion_level_source = occlusion_name_to_level_mapping[occlusion_level_source]
                occlusion_level = min(occlusion_level_target, occlusion_level_source)
                occlusion_level_merged = occlusion_level_to_name_mapping[occlusion_level]

            # store average values in object_3d_south
            object_3d_target["object_data"]["cuboid"]["val"] = [
                position_3d_merged[0],
                position_3d_merged[1],
                position_3d_merged[2],
                quaternion_target[0],
                quaternion_target[1],
                quaternion_target[2],
                quaternion_target[3],
                dimensions_3d_merged[0],
                dimensions_3d_merged[1],
                dimensions_3d_merged[2],
            ]
            object_3d_target["object_data"]["cuboid"]["attributes"]["num"] = [
                {"name": "num_points", "val": num_points_merged},
                {"name": "score", "val": score_merged},
                {"name": "number_of_trailers", "val": number_of_trailers_merged},
            ]

            # check if attribute exists
            attribute = VisualizationUtils.get_attribute_by_name(
                object_3d_source["object_data"]["cuboid"]["attributes"]["text"],
                "sensor_id",
            )

            object_3d_target["object_data"]["cuboid"]["attributes"]["text"] = []
            object_3d_target["object_data"]["cuboid"]["attributes"]["text"].append(
                {"name": "occlusion_level", "val": occlusion_level_merged}
            )
            if attribute is not None:
                object_3d_target["object_data"]["cuboid"]["attributes"]["text"].append(
                    {"name": "sensor_id", "val": "s110_lidar_ouster_south_and_s110_lidar_ouster_north"}
                )
            else:
                object_3d_target["object_data"]["cuboid"]["attributes"]["text"].append(
                    {"name": "sensor_id", "val": "s110_lidar_ouster_south_and_s110_lidar_ouster_north"}
                )

        # handle unmatched objects
        objects_3d_source_ids_hex = [objects_3d_source_ids_hex[x] for x in unmatched_source_ids]

        frame_obj_source = next(iter(json_label_target["openlabel"]["frames"].items()))[1]
        for id_hex, unmatch_id_source in zip(objects_3d_source_ids_hex, unmatched_source_ids):
            object_3d_source = objects_3d_source_list[unmatch_id_source]

            position_3d_source = np.array(object_3d_source["object_data"]["cuboid"]["val"])[:3]

            if not same_origin:
                position_3d_source = np.matmul(
                    transformation_matrix,
                    np.array(
                        [
                            position_3d_source[0],
                            position_3d_source[1],
                            position_3d_source[2],
                            1,
                        ]
                    ),
                )
            object_3d_source["object_data"]["cuboid"]["val"][0] = position_3d_source[0]
            object_3d_source["object_data"]["cuboid"]["val"][1] = position_3d_source[1]
            object_3d_source["object_data"]["cuboid"]["val"][2] = position_3d_source[2]

            quaternion_source = np.array(object_3d_source["object_data"]["cuboid"]["val"])[3:7]
            roll_source, pitch_source, yaw_source = Rotation.from_quat(quaternion_source).as_euler(
                "xyz", degrees=True
            )

            if not same_origin:
                delta_yaw = 177.90826842 - 163.58774077
                yaw_source += delta_yaw

            # convert back to quaternion
            quaternion_source = R.from_euler(
                seq="xyz",
                angles=[roll_source, pitch_source, yaw_source],
                degrees=True,
            ).as_quat()

            # store new rotation
            object_3d_source["object_data"]["cuboid"]["val"][3] = quaternion_source[0]
            object_3d_source["object_data"]["cuboid"]["val"][4] = quaternion_source[1]
            object_3d_source["object_data"]["cuboid"]["val"][5] = quaternion_source[2]
            object_3d_source["object_data"]["cuboid"]["val"][6] = quaternion_source[3]
            frame_obj_source["objects"][id_hex] = object_3d_source

        if "s110_lidar_ouster_south" in boxes_file_name_target:
            output_file_name = boxes_file_name_target.replace(
                "s110_lidar_ouster_south", "s110_lidar_ouster_south_and_north_merged"
            )
        if "s110_lidar_ouster_north" in boxes_file_name_target:
            output_file_name = boxes_file_name_target.replace(
                "s110_lidar_ouster_north", "s110_lidar_ouster_north_and_south_merged"
            )
        if "s110_camera_basler_south1_8mm" in boxes_file_name_target:
            output_file_name = boxes_file_name_target.replace(
                "s110_camera_basler_south1_8mm",
                "s110_camera_basler_south1_8mm_and_south2_8mm_merged",
            )
        # add file name from south lidar and north lidar to frame properties
        if "frame_properties" in frame_obj_source and "point_cloud_file_name" in frame_obj_source["frame_properties"]:
            del frame_obj_source["frame_properties"]["point_cloud_file_name"]
            frame_obj_source["frame_properties"]["point_cloud_file_names"] = [
                boxes_file_name_target.replace(".json", ".pcd"),
                boxes_file_name_source.replace(".json", ".pcd"),
            ]
        store_boxes_in_open_label(json_label_target, output_folder_path_fused_boxes, output_file_name)
        frame_id += 1

    print("num matches: ", num_matches)
