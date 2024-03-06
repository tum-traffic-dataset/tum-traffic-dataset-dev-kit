import json
import numpy as np
from pathlib import Path
import os
from argparse import ArgumentParser

# map object class name to MS COCO IDs
from src.utils.vis_utils import class_name_to_id_mapping


# Conversion for A9 dataset (release 0 and 1) to Yolo format
# Example:
# python conversion_openlabel_to_yolo.py --input_folder_path_labels a9_dataset_r00_s01/_labels --output_folder_path_labels a9_dataset_r00_s01/_labels_yolo


# MS Coco class IDs
def get_id_by_class_name(object_class):
    object_class_id = -1
    if object_class == "pedestrian":
        object_class_id = 0
    if object_class == "bicycle":
        object_class_id = 1
    if object_class == "car":
        object_class_id = 2
    if object_class == "motorcycle":
        object_class_id = 3
    if object_class == "bus":
        object_class_id = 5
    if object_class == "truck":
        object_class_id = 7
    return object_class_id


def convert_file(input_folder_path_labels, output_folder_path_labels, filename, export_format):
    open(os.path.join(output_folder_path_labels, filename + ".txt"), "w").close()

    with open(os.path.join(input_folder_path_labels, filename + ".json"), "r") as input_file:
        data = json.load(input_file)

    if "openlabel" in data:
        # converting A9 Release 1 or higher to Yolo
        for frame_id, frame_obj in data["openlabel"]["frames"].items():
            for object_id, object_json in frame_obj["objects"].items():
                object_class = object_json["object_data"]["type"].upper()
                if "bbox" in object_json["object_data"]:
                    bbox = object_json["object_data"]["bbox"][0]["val"]  # x_center, y_center, width, height
                else:
                    # calculate bbox from 2D keypoints
                    keypoints = object_json["object_data"]["keypoints_2d"]["attributes"]["points2d"]["val"]
                    bbox = [0, 0, 0, 0]
                    # iterate over 8 keypoints
                    keypoints_xy = []
                    for keypoint in keypoints:
                        x = keypoint["point2d"]["val"][0]
                        y = keypoint["point2d"]["val"][1]
                        keypoints_xy.append([x, y])
                    # calculate bbox
                    x_min = np.amin(np.array(keypoints_xy)[:, 0])
                    x_max = np.amax(np.array(keypoints_xy)[:, 0])
                    y_min = np.amin(np.array(keypoints_xy)[:, 1])
                    y_max = np.amax(np.array(keypoints_xy)[:, 1])
                    bbox[0] = int((x_min + x_max) / 2.0)
                    bbox[1] = int((y_min + y_max) / 2.0)
                    bbox[2] = int(x_max - x_min)
                    bbox[3] = int(y_max - y_min)

                x_center = bbox[0]
                y_center = bbox[1]
                width = bbox[2]
                height = bbox[3]

                if int(x_center - width / 2.0) < 0:
                    # update x_center, in case it is negative
                    x_max = int(x_center + width / 2.0)
                    x_center = int(x_max / 2.0)
                    width = x_max

                if int(x_center + width / 2.0) > 1920:
                    # update x_center, in case it is larger than image width
                    x_min = int(x_center - width / 2.0)
                    width = 1920 - x_min
                    x_center = x_min + int(width / 2.0)

                if int(y_center - height / 2.0) < 0:
                    # update y_center, in case it is negative
                    y_max = int(y_center + height / 2.0)
                    y_center = int(y_max / 2.0)
                    height = y_max

                if int(y_center + height / 2.0) > 1200:
                    # update y_center, in case it is larger than image height
                    y_min = int(y_center - height / 2.0)
                    height = 1200 - y_min
                    y_center = y_min + int(height / 2.0)

                if export_format == "transfer_learning":  # transfer learning with MS Coco classes
                    object_class_id = get_id_by_class_name(object_json["object_data"]["type"].lower())
                elif export_format == "custom_training":
                    object_class_id = class_name_to_id_mapping[object_class]
                write_line(output_folder_path_labels, filename, object_class_id, x_center, y_center, height, width)
    else:
        # converting A9 Release 0 to Yolo
        for labels in data["labels"]:
            print("---------")
            A = np.empty((0, 2), np.float64)
            object_class = labels["category"]
            print(object_class)
            for box3d_projected in labels["box3d_projected"]:
                x = np.float64(labels["box3d_projected"][box3d_projected][0])
                y = np.float64(labels["box3d_projected"][box3d_projected][1])
                print(x, y)
                A = np.append(A, [[x, y]], axis=0)

            print(np.matrix(A))

            x_min = np.amin(A[:, 0])
            x_max = np.amax(A[:, 0])
            y_min = np.amin(A[:, 1])
            y_max = np.amax(A[:, 1])
            print("x_min, x_max, y_min, y_max: ", x_min, x_max, y_min, y_max)

            width = x_max - x_min
            height = y_max - y_min
            print("Width, height: ", width, height)

            x_center = x_min + width / 2
            y_center = y_min + height / 2
            print("x_midpoint, y_midpoint: ", x_center, y_center)

            object_class = object_class.lower()
            object_id = get_id_by_class_name(object_class)

            if object_id == -1:
                continue

            print("object_class_id: ", object_id)

            write_line(filename, object_id, x_center, y_center, height, width)

            print("-*-*-*-*-")


def write_line(output_folder_path_labels, filename, object_id, x_center, y_center, height, width):
    new_line = (
            str(object_id)
            + " "
            + str(x_center / 1920)
            + " "
            + str(y_center / 1200)
            + " "
            + str(width / 1920)
            + " "
            + str(height / 1200)
            + "\n"
    )
    with open(os.path.join(output_folder_path_labels, filename + ".txt"), "a") as output_file:
        output_file.write(new_line)


def convert_a9_to_mscoco(input_folder_path_labels, output_folder_path_labels, export_format):
    for input_file_path_label in Path(input_folder_path_labels).rglob("*.json"):
        print("Processing file ", input_file_path_label.name)
        input_file_name_no_suffix = os.path.splitext(input_file_path_label.name)[0]
        convert_file(input_folder_path_labels, output_folder_path_labels, input_file_name_no_suffix, export_format)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder_path_labels", type=str, default="input_labels")
    parser.add_argument("--output_folder_path_labels", type=str, default="output_labels")
    parser.add_argument(
        "--export_format",
        type=str,
        default="custom_training",
        help="Specify the export format of the class IDs. Possible values: [custom_training, transfer_learning]. custom_training: Use IDs from A9 dataset to train from scratch. transfer_learning: Use IDs from MS COCO dataset to do transfer learning.",
    )
    args = parser.parse_args()
    input_folder_path_labels = args.input_folder_path_labels
    output_folder_path_labels = args.output_folder_path_labels
    export_format = args.export_format
    if not os.path.exists(output_folder_path_labels):
        Path(output_folder_path_labels).mkdir(parents=True, exist_ok=True)
    convert_a9_to_mscoco(input_folder_path_labels, output_folder_path_labels, export_format)
