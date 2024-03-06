import sys
import os
import cv2
import glob
import numpy as np
import json
import argparse


# This script undistorts/rectifies 2D bounding boxes.
# Usage:
#           python a9-dev-kit/undistort_points.py --input_folder_path_bounding_boxes <INPUT_FOLDER_PATH_BOUNDING_BOXES> \
#                                                 --file_path_calibration_parameter <FILE_PATH_CALIBRATION_PARAMETER> \
#                                                 --output_folder_path_bounding_boxes <OUTPUT_FOLDER_PATH_BOUNDING_BOXES>


def load_calibration_parameters(file_path_calibration_parameter):
    print("Loading camera parameters from file:", file_path_calibration_parameter)
    return json.load(
        open(
            file_path_calibration_parameter,
        )
    )


def undistort_point(pt, K, dist_coeffs):
    # see link for reference on the equation
    # http://opencv.willowgarage.com/documentation/camera_calibration_and_3d_reconstruction.html
    u, v = pt
    k1, k2, p1, p2, k3 = dist_coeffs
    u0 = K[0][2]  # cx
    v0 = K[1][2]  # cy
    fx = K[0][0]
    fy = K[1][1]
    _fx = 1.0 / fx
    _fy = 1.0 / fy
    y = (v - v0) * _fy
    x = (u - u0) * _fx
    r = np.sqrt(x ** 2 + y ** 2)
    u_undistort = (x * (1 + (k1 * r ** 2) + (k2 * r ** 4) + (k3 * r ** 6))) + 2 * p1 * x * y + p2 * (
            r ** 2 + 2 * x ** 2)
    v_undistort = (y * (1 + (k1 * r ** 2) + (k2 * r ** 4) + (k3 * r ** 6))) + 2 * p2 * y * x + p1 * (
            r ** 2 + 2 * y ** 2)
    x_undistort = fx * u_undistort + u0
    y_undistort = fy * v_undistort + v0
    return x_undistort, y_undistort


def undistort_points(points, calib_params, use_optimal_intrinsic_camera_matrix):
    if use_optimal_intrinsic_camera_matrix:
        optimal_intrinsic_camera_matrix = np.array(calib_params["optimal_intrinsic_camera_matrix"])
    else:
        optimal_intrinsic_camera_matrix = np.array(calib_params["intrinsic_camera_matrix"])

    points_undistorted = cv2.undistortPoints(
        points,
        np.array(calib_params["intrinsic_camera_matrix"])[:3, :3],
        np.array(calib_params["dist_coefficients"], dtype=np.float32),
        None,
        optimal_intrinsic_camera_matrix[:3, :3],
    )
    # convert from (4,1,2) to (4,2)
    points_undistorted = points_undistorted.reshape(-1, 2)
    return points_undistorted


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Argument Parser")
    argparser.add_argument(
        "-i",
        "--input_folder_path_bounding_boxes",
        default="a9_dataset/r01_s01/_labels/s040_camera_basler_north_16mm",
        help="Input folder path to bounding boxes. Default: a9_dataset/r01_s01/_labels/s040_camera_basler_north_16mm",
    )
    argparser.add_argument(
        "-l",
        "--file_path_calibration_parameter",
        default="a9_dataset/r00_s00/_calibration/s040_camera_basler_north_16mm.json",
        help="File path to calibration parameter file (json). Default: a9_dataset/r01_s01/_calibration/s040_camera_basler_north_16mm.json",
    )
    argparser.add_argument(
        "-o",
        "--output_folder_path_bounding_boxes",
        help="Output folder path to undistorted/rectified bounding boxes. Default: a9_dataset/r01_s01/_labels_undistorted/s040_camera_basler_north_16mm",
    )
    args = argparser.parse_args()
    input_folder_path_bounding_boxes = args.input_folder_path_bounding_boxes
    file_path_calibration_parameter = args.file_path_calibration_parameter
    output_folder_path_bounding_boxes = args.output_folder_path_bounding_boxes

    # load calibration parameters
    calib_params = {}
    calib_params = load_calibration_parameters(file_path_calibration_parameter)
    input_file_paths_bounding_boxes = sorted(glob.glob(input_folder_path_bounding_boxes + "/*.json"))

    if not os.path.exists(output_folder_path_bounding_boxes):
        os.makedirs(output_folder_path_bounding_boxes)

    if "8mm" in input_folder_path_bounding_boxes or "50mm" in input_folder_path_bounding_boxes:
        use_optimal_intrinsic_camera_matrix = True
    else:
        use_optimal_intrinsic_camera_matrix = False

    for bounding_boxes_filepath in input_file_paths_bounding_boxes:
        # read openlabel json file
        labels_json = json.load(open(bounding_boxes_filepath))

        # here the bounding boxes get undistorted using the opencv undistortPoints method
        for frame_id, frame_obj in labels_json["openlabel"]["frames"].items():
            # iterate over all objects
            for object_id, label in frame_obj["objects"].items():
                bounding_box = np.array(label["object_data"]["bbox"][0]["val"])
                # create a 4x2 2D array with 4 corner points of the bounding box
                x_center = bounding_box[0]
                y_center = bounding_box[1]
                width = bounding_box[2]
                height = bounding_box[3]
                bounding_box = np.array(
                    [
                        [x_center - width / 2, y_center - height / 2],
                        [x_center + width / 2, y_center - height / 2],
                        [x_center + width / 2, y_center + height / 2],
                        [x_center - width / 2, y_center + height / 2],
                    ]
                )
                bounding_box_undistorted = []
                bounding_box = bounding_box.astype(float)
                # make bounding_boxes 1x4x2
                bounding_box = bounding_box.reshape(1, -1, 2)
                undistorted_bounding_boxes = undistort_points(bounding_box, calib_params,
                                                              use_optimal_intrinsic_camera_matrix)
                x_min_undistorted = undistorted_bounding_boxes[0][0]
                y_min_undistorted = undistorted_bounding_boxes[0][1]
                x_max_undistorted = undistorted_bounding_boxes[1][0]
                y_max_undistorted = undistorted_bounding_boxes[2][1]
                width_undistorted = x_max_undistorted - x_min_undistorted
                height_undistorted = y_max_undistorted - y_min_undistorted
                # # save undistorted bounding boxes to json file
                x_center_undistorted = x_min_undistorted + width / 2
                y_center_undistorted = y_min_undistorted + height / 2
                label["object_data"]["bbox"][0]["val"] = [x_center_undistorted, y_center_undistorted, width_undistorted,
                                                          height_undistorted]

        # get file name from file path
        filename = os.path.basename(bounding_boxes_filepath)
        # save undistorted bounding boxes to json file
        with open(os.path.join(output_folder_path_bounding_boxes, filename), "w") as f:
            json.dump(labels_json, f, indent=4, sort_keys=True)
