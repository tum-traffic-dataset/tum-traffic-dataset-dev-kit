import argparse
import os
import json
import numpy as np

from src.utils.vis_utils import VisualizationUtils

if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_labels_train",
        type=str,
        help="Path to train labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_val",
        type=str,
        help="Path to val labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sampled",
        type=str,
        help="Path to test sampled labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sequence",
        type=str,
        help="Path to test sequence labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_south",
        type=str,
        help="Path to r02_s01 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_north",
        type=str,
        help="Path to r02_s01 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_south",
        type=str,
        help="Path to r02_s02 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_north",
        type=str,
        help="Path to r02_s02 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_south",
        type=str,
        help="Path to r02_s03 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_north",
        type=str,
        help="Path to r02_s03 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_south",
        type=str,
        help="Path to r02_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_north",
        type=str,
        help="Path to r02_s04 north lidar labels",
        default="",
    )
    args = arg_parser.parse_args()

    input_folder_paths_all = []
    if args.input_folder_path_labels_train:
        input_folder_paths_all.append(args.input_folder_path_labels_train)
    if args.input_folder_path_labels_val:
        input_folder_paths_all.append(args.input_folder_path_labels_val)
    if args.input_folder_path_labels_test_sampled:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sampled)
    if args.input_folder_path_labels_test_sequence:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sequence)

    if args.input_folder_path_labels_sequence_s01_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_south)
    if args.input_folder_path_labels_sequence_s01_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_north)
    if args.input_folder_path_labels_sequence_s02_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_south)
    if args.input_folder_path_labels_sequence_s02_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_north)
    if args.input_folder_path_labels_sequence_s03_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_south)
    if args.input_folder_path_labels_sequence_s03_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_north)
    if args.input_folder_path_labels_sequence_s04_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_south)
    if args.input_folder_path_labels_sequence_s04_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_north)
    num_labels_2d_boxes = 0
    num_labels_2d_boxes_vehicle = 0
    num_labels_2d_masks = 0
    num_labels_3d = 0
    num_attributes = 0
    num_unique_labels = []
    num_labels = 0
    for input_folder_path_labels in input_folder_paths_all:
        for label_file_name in sorted(os.listdir(input_folder_path_labels)):
            num_labels_2d_boxes_current_file = 0
            num_labels_3d_boxes_current_file = 0
            json_file = open(
                os.path.join(input_folder_path_labels, label_file_name),
            )
            json_data = json.load(json_file)
            if "labels" in json_data:
                num_labels_2d_boxes += len(json_data["labels"])
                continue
            # num_labels += len(json_data["labels"])
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                for object_track_id, object_json in frame_obj["objects"].items():
                    object_data = object_json["object_data"]

                    # count 2d boxes
                    if "bbox" in object_data and len(object_data["bbox"]) > 0:
                        num_labels_2d_boxes += len(object_data["bbox"])
                        num_labels_2d_boxes_current_file += len(object_data["bbox"])
                        # count bboxes with sensor_id = vehicle_camera_basler_16mm
                        for bbox in object_data["bbox"]:
                            if "attributes" in bbox:
                                sensor_id_attribute = VisualizationUtils.get_attribute_by_name(
                                    bbox["attributes"]["text"], "sensor_id"
                                )
                                if sensor_id_attribute is not None and sensor_id_attribute["val"] == "vehicle_camera_basler_16mm":
                                    num_labels_2d_boxes_vehicle += 1
                    # count 2d masks
                    if "poly2d" in object_data and len(object_data["poly2d"]) > 0:
                        num_labels_2d_masks += 1
                    # count attributes
                    if "cuboid" in object_data and "attributes" in object_data["cuboid"]:
                        if "text" in object_data["cuboid"]["attributes"]:
                            num_attributes += len(object_data["cuboid"]["attributes"]["text"])
                        if "num" in object_data["cuboid"]["attributes"]:
                            num_attributes += len(object_data["cuboid"]["attributes"]["num"])
                        if "boolean" in object_data["cuboid"]["attributes"]:
                            num_attributes += len(object_data["cuboid"]["attributes"]["boolean"])
                    # count 3d boxes
                    # invalid cuboid: [
                    #                                     0.0,
                    #                                     0.0,
                    #                                     0.0,
                    #                                     0.0,
                    #                                     0.0,
                    #                                     0.0,
                    #                                     1.0,
                    #                                     0.0,
                    #                                     0.0,
                    #                                     0.0
                    #                                 ]
                    # if the sum of all array elements is 1 then the cuboid is invalid
                    if "cuboid" in object_data:
                        if np.all(np.array(object_data["cuboid"]["val"]) == 0) or np.sum(
                                np.array(object_data["cuboid"]["val"])) == 1:
                            continue
                        else:
                            num_labels_3d += 1
                            num_labels_3d_boxes_current_file += 1
                for object_key in frame_obj["objects"].keys():
                    if object_key not in num_unique_labels:
                        num_unique_labels.append(object_key)

            # statistics per file
            if num_labels_2d_boxes_current_file != num_labels_3d_boxes_current_file:
                print("file name: ", label_file_name)
                print("num_labels_2d_boxes_current_file:", str(num_labels_2d_boxes_current_file))
                print("num_labels_3d_boxes_current_file:", str(num_labels_3d_boxes_current_file))

    print("num labels 2D boxes total:", str(num_labels_2d_boxes))
    print("num labels 2D boxes vehicle total:", str(num_labels_2d_boxes_vehicle))
    print("num labels 2D masks total:", str(num_labels_2d_masks))
    print("num labels 3D total:", str(num_labels_3d))
    print("num unique labels total:", str(len(num_unique_labels)))
    print("num labels total:", str(num_labels))
    print("num attributes total:", str(num_attributes))

    # mono3d s110_camera_basler_south1_8mm
    # num detections 2D total: 1870 (yolov7)
    # num detections 3D total: 1790 (detection_processing)
    # num detections 3D total: 1082 (mono3d)

    # mono3d s110_camera_basler_south2_8mm
    # num detections 2D total: 1622 (yolov7)
    # num detections 3D total: 1563 (detection_processing)
    # num detections 3D total: 1068 (mono3d)

    # full split lidar labels
    # train
    # num labels 2D total: 28086
    # num labels 3D total: 28086
    # num unique labels total: 437
    # val
