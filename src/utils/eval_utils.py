import argparse

import numpy as np


def compute_split_parts(num_samples, num_parts):
    part_samples = num_samples // num_parts
    remain_samples = num_samples % num_parts
    if part_samples == 0:
        return [num_samples]
    if remain_samples == 0:
        return [part_samples] * num_parts
    else:
        return [part_samples] * num_parts + [remain_samples]


def overall_filter(boxes, level):
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    if len(boxes) == 0:
        return ignore

    # calculate euclidian distance
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if level == 0:
        flag = dist < 64
    elif level == 1:  # 0-40m
        flag = dist < 40
    elif level == 2:  # 40-50m
        flag = (dist >= 40) & (dist < 50)
    elif level == 3:  # 50m-inf
        # TODO: temp crop labels at 64 m
        flag = (dist >= 50) & (dist < 64)
    else:
        assert False, "level < 4 for overall & distance metric, found level %s" % (str(level))

    ignore[flag] = False
    return ignore


def overall_distance_filter(boxes, level):
    # check if boxes are empty
    if len(boxes) == 0:
        print("shape of boxes", boxes.shape)
        return np.ones(boxes.shape[0], dtype=bool)
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if level == 0:
        flag = np.ones(boxes.shape[0], dtype=bool)
    elif level == 1:  # 0-40m
        flag = dist < 40
    elif level == 2:  # 40-50m
        flag = (dist >= 40) & (dist < 50)
    elif level == 3:  # 50m-inf
        # TODO: temp crop labels at 64 m
        flag = (dist >= 50) & (dist < 64)
    else:
        assert False, "level < 4 for overall & distance metric, found level %s" % (str(level))

    ignore[flag] = False
    return ignore


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Evaluating A9-Dataset R2 Release")
    argparser.add_argument("-gt", "--folder_path_ground_truth", type=str, help="Ground truth folder path.")
    argparser.add_argument("-p", "--folder_path_predictions", type=str, help="Predictions folder path.")
    argparser.add_argument(
        "--object_min_points", type=int, default=5, help="Minimum point per object before being considered."
    )
    argparser.add_argument(
        "--use_superclasses",
        action="store_true",
        help="Group single classes into 3 groups [Vehicle=(Car,Truck, Trailer, Van, Bus, Emergency_Vehicle, Other), Bicycle=(Bicycle, Motorcycle), Pedestrian=(Pedestrian).]",
    )
    argparser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="Prediction type of input detections. Can be [lidar3d_unsupervised, lidar3d_supervised, mono3d, multi3d]",
    )
    argparser.add_argument(
        "--prediction_format",
        type=str,
        default="openlabel",
        help="Prediction format of detections. Can be [openlabel, kitti]",
    )
    argparser.add_argument("--use_ouster_lidar_only", action="store_true", help="Use only Ouster lidar for evaluation.")
    argparser.add_argument(
        "--camera_id",
        type=str,
        default=None,
        help="Provide the camera ID you want to use for evaluation. Possible values are: [s110_camera_basler_south1_8mm, s110_camera_basler_south2_8mm]",
    )
    argparser.add_argument("--file_path_calibration_data", default="", help="File path to calibration data.")

    args = argparser.parse_args()
    return args
