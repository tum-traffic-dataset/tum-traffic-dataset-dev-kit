import sys
import os
import cv2
import glob
import numpy as np
import json
import argparse


# This script undistorts/rectifies camera images. It transforms an image to compensate for radial and tangential lens distortion.
# Usage:
#           python a9-dev-kit/undistort_images.py --input_folder_path_images <INPUT_FOLDER_PATH_IMAGES> \
#                                                 --file_path_calibration_parameter <FILE_PATH_CALIBRATION_PARAMETER> \
#                                                 --output_folder_path_images <OUTPUT_FOLDER_PATH_IMAGES>
# Example:
#           python a9-dev-kit/undistort_images.py --input_folder_path_images a9_dataset/r00_s00/_images/s040_camera_basler_north_16mm \
#                                                 --file_path_calibration_parameter a9_dataset/r00_s00/_calibration/s040_camera_basler_north_16mm.json \
#                                                 --output_folder_path_images a9_dataset/r00_s00/_images_undistorted


def load_calibration_parameters(file_path_calibration_parameter):
    print("Loading camera parameters from file: ", file_path_calibration_parameter)
    return json.load(
        open(
            file_path_calibration_parameter,
        )
    )


def undistort_image(image, calib_params, use_optimal_intrinsic_camera_matrix):
    if image.shape[0] == calib_params["image_height"] and image.shape[1] == calib_params["image_width"]:
        if use_optimal_intrinsic_camera_matrix:
            optimal_intrinsic_camera_matrix = np.array(calib_params["optimal_intrinsic_camera_matrix"])
        else:
            optimal_intrinsic_camera_matrix = np.array(calib_params["intrinsic_camera_matrix"])

        img_undistorted = cv2.undistort(
            image,
            np.array(calib_params["intrinsic_camera_matrix"])[:3, :3],
            np.array(calib_params["dist_coefficients"]),
            None,
            optimal_intrinsic_camera_matrix[:3, :3],
        )
        return img_undistorted
    else:
        print("Error. image height/width does not match image height/width in calib file. Exiting...")
        sys.exit(0)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="VizLabel Argument Parser")
    argparser.add_argument(
        "-i",
        "--input_folder_path_images",
        default="a9_dataset/r01_s01/_images/s040_camera_basler_north_16mm",
        help="Input folder path of images. Default: a9_dataset/r01_s01/_images/s040_camera_basler_north_16mm",
    )
    argparser.add_argument(
        "-l",
        "--file_path_calibration_parameter",
        default="a9_dataset/r00_s00/_calibration/s040_camera_basler_north_16mm.json",
        help="File path to calibration parameter file (json). Default: a9_dataset/r01_s01/_calibration/s040_camera_basler_north_16mm.json",
    )
    argparser.add_argument(
        "-o",
        "--output_folder_path_images",
        help="Output folder path to undistorted/rectified camera images. Default: a9_dataset/r01_s01/_images_undistorted/s040_camera_basler_north_16mm",
    )
    args = argparser.parse_args()
    input_folder_path_images = args.input_folder_path_images
    file_path_calibration_parameter = args.file_path_calibration_parameter
    output_folder_path_images = args.output_folder_path_images

    # load calibration parameters
    calib_params = {}
    calib_params = load_calibration_parameters(file_path_calibration_parameter)
    input_file_paths_images = sorted(
        glob.glob(input_folder_path_images + "/*.jpg") + glob.glob(input_folder_path_images + "/*.png")
    )

    if not os.path.exists(output_folder_path_images):
        os.makedirs(output_folder_path_images)

    file_name_calib_params = file_path_calibration_parameter.split("/")[-1]
    if "8mm" in file_name_calib_params or "50mm" in file_name_calib_params:
        use_optimal_intrinsic_camera_matrix = True
    else:
        use_optimal_intrinsic_camera_matrix = False

    for image_filepath in input_file_paths_images:
        # Capture frame-by-frame
        img = cv2.imread(image_filepath)
        img_filename = image_filepath.split("/")[-1]
        # here the image frame gets undistorted using the opencv undistort method
        frame_undistorted = undistort_image(img, calib_params, use_optimal_intrinsic_camera_matrix)

        # Display the resulting frame
        cv2.imwrite(os.path.join(output_folder_path_images, img_filename), frame_undistorted)
