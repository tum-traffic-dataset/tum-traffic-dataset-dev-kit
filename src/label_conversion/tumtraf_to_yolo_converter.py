import json
import numpy as np
from pathlib import Path
import os
from argparse import ArgumentParser

# Conversion for A9 dataset (release 0 and 1) to Yolo format
# Example:
# python conversion_openlabel_to_yolo.py --input_folder_path_labels a9_dataset_r00_s01/_labels --output_folder_path_labels a9_dataset_r00_s01/_labels_yolo


# MS Coco class IDs
#if object_class == "pedestrian":
#    object_class_id = 0
#if object_class == "bicycle":
#    object_class_id = 1
#if object_class == "car":
#    object_class_id = 2
#if object_class == "motorcycle":
#    object_class_id = 3
#if object_class == "bus":
#    object_class_id = 5
#if object_class == "truck":
#    object_class_id = 7

IMG_WIDTH = 640
IMG_HEIGHT = 480

def get_id_by_class_name(object_class):
    object_class = object_class.lower()
    object_class_id = -1
    if object_class == "pedestrian":
        object_class_id = 0
    if object_class == "bicycle":
        object_class_id = 1
    if object_class == "car":
        object_class_id = 2
    if object_class == "van":
        object_class_id = 2
    if object_class == "motorcycle":
        object_class_id = 3
    if object_class == "bus":
        object_class_id = 4
    if object_class == "truck":
        object_class_id = 5
    if object_class == "trailer":
        object_class_id = 6
    return object_class_id


def convert_file(input_folder_path_labels, output_folder_path_labels, filename, release):
    object_class_id = -1

    open(os.path.join(output_folder_path_labels, filename + ".txt"), "w").close()

    with open(os.path.join(input_folder_path_labels, filename + ".json"), "r") as input_file:
        data = json.load(input_file)

    if release == "2_openlabel_correct": # With a9_dataset_r02_s04 respectively r01_s09
        for frame_id, frames_content in data["openlabel"]["frames"].items(): #["0"]["objects"].items():
            objects = frames_content["objects"]
            for object_id, object_data in objects.items():
                bbox = object_data["object_data"]["bbox"][0]["val"]
                object_class_name = object_data["object_data"]["type"]

                x_center = bbox[0]
                y_center = bbox[1]
                width = bbox[2]
                height = bbox[3]
                height, width, x_center, y_center = check_image_boundaries(height, width, x_center, y_center)
                print("object_class_name, x_center, y_center, width, height ",
                      object_class_name, x_center, y_center, width, height)

                object_class_id = get_id_by_class_name(object_class_name)
                print("object_class_id: ", object_class_id)
                if object_class_id == -1:
                    print("Skip unknown class.")
                    continue

                write_line_release_normalized(output_folder_path_labels, filename, object_class_id, x_center, y_center,
                                              height, width)

    elif release == "2":
        # Converting A9 Releases in Open Label Format to Yolo
        for object_id, object_json in data["frames"]["objects"].items():
            print("object_id: ", object_id)
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
            height, width, x_center, y_center = check_image_boundaries(height, width, x_center, y_center)

            object_class_name = object_json["object_data"]["type"]

            print("object_class_name, x_center, y_center, width, height ",
                  object_class_name, x_center, y_center, width, height)

            object_class_id = get_id_by_class_name(object_class_name)
            print("object_class_id: ", object_class_id)
            if object_class_id == -1:
                print("Skip unknown class.")
                continue

            write_line_release_normalized(output_folder_path_labels, filename, object_class_id, x_center, y_center,
                                          height, width)

    elif release == "1" or release =="0":
        # Converting A9 Releases in previous format to Yolo
        for labels in data["labels"]:
            print("---------")
            A = np.empty((0, 2), np.float64)
            object_class_name = labels["category"]

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

            width = x_max - x_min
            height = y_max - y_min
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            if release == "0":
                x_center = x_center * IMG_WIDTH
                y_center = y_center * IMG_HEIGHT
                width = width * IMG_WIDTH
                height = height * IMG_HEIGHT
            height, width, x_center, y_center = check_image_boundaries(height, width, x_center, y_center)

            print("object_class_name, x_center, y_center, width, height ",
                  object_class_name, x_center, y_center, width, height)

            object_class_id = get_id_by_class_name(object_class_name)
            print("object_class_id: ", object_class_id)
            if object_class_id == -1:
                print("Skip unknown class.")
                continue

            write_line_release_normalized(output_folder_path_labels, filename, object_class_id, x_center, y_center,
                                          height, width)


def check_image_boundaries(height, width, x_center, y_center):
    if int(x_center - width / 2.0) < 0:
        # update x_center, in case it is negative
        x_max = int(x_center + width / 2.0)
        x_center = int(x_max / 2.0)
        width = x_max
    if int(x_center + width / 2.0) > IMG_WIDTH:
        # update x_center, in case it is larger than image width
        x_min = int(x_center - width / 2.0)
        width = IMG_WIDTH - x_min
        x_center = x_min + int(width / 2.0)
    if int(y_center - height / 2.0) < 0:
        # update y_center, in case it is negative
        y_max = int(y_center + height / 2.0)
        y_center = int(y_max / 2.0)
        height = y_max
    if int(y_center + height / 2.0) > IMG_HEIGHT:
        # update y_center, in case it is larger than image height
        y_min = int(y_center - height / 2.0)
        height = IMG_HEIGHT - y_min
        y_center = y_min + int(height / 2.0)
    return height, width, x_center, y_center


def write_line_release_normalized(output_folder_path_labels, filename, object_id, x_center, y_center, height, width):
    new_line = (
            str(object_id) + " "
            + str(x_center / IMG_WIDTH) + " "
            + str(y_center / IMG_HEIGHT) + " "
            + str(width / IMG_WIDTH) + " "
            + str(height / IMG_HEIGHT) + "\n"
    )
    with open(os.path.join(output_folder_path_labels, filename + ".txt"), "a") as output_file:
        output_file.write(new_line)


def convert_a9_to_yolo(input_folder_path_labels, output_folder_path_labels, release):
    for input_file_path_label in Path(input_folder_path_labels).rglob("*.json"):
        print("Processing file ", input_file_path_label.name)
        input_file_name_no_suffix = os.path.splitext(input_file_path_label.name)[0]
        convert_file(input_folder_path_labels, output_folder_path_labels, input_file_name_no_suffix, release)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder_path_labels", type=str, default="input_labels")
    parser.add_argument("--output_folder_path_labels", type=str, default="output_labels")

    args = parser.parse_args()
    input_folder_path_labels = args.input_folder_path_labels
    output_folder_path_labels = args.output_folder_path_labels
    release = "2_openlabel_correct" #0, 1, 2, 2_openlabel_correct


    if not os.path.exists(output_folder_path_labels):
        Path(output_folder_path_labels).mkdir(parents=True, exist_ok=True)
    convert_a9_to_yolo(input_folder_path_labels, output_folder_path_labels, release)
