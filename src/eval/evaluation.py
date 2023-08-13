"""
Evaluation
"""
import glob
import json
import math
import os

import numpy as np
import numba
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from src.utils.iou_utils import rotate_iou_gpu_eval
from src.utils.rotate_iou_cpu_eval import rotate_iou_cpu_eval
from src.utils.eval_utils import compute_split_parts, overall_filter, parse_arguments

##################################
# Evaluation Script for A9 3D Object Detection
##################################
#
# Example usage:
# python evaluation.py --camera_id <CAMERA_ID> --folder_path_ground_truth /path/to/ground_truth --folder_path_predictions /path/to/predictions --object_min_points 5 [--use_superclasses] --prediction_type lidar3d_supervised --prediction_format openlabel --use_ouster_lidar_only

import sys
from pathlib import Path

from src.utils.perspective import parse_perspective

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

iou_threshold_dict = {
    "CAR": 0.1,
    "TRUCK": 0.1,
    "TRAILER": 0.1,
    "VAN": 0.1,
    "MOTORCYCLE": 0.1,
    "BUS": 0.1,
    "PEDESTRIAN": 0.1,
    "BICYCLE": 0.1,
    "EMERGENCY_VEHICLE": 0.1,
    "OTHER": 0.1,
}

superclass_iou_threshold_dict = {"VEHICLE": 0.1, "PEDESTRIAN": 0.1, "BICYCLE": 0.1}  # 0.7  # 0.3  # 0.5


def get_evaluation_results(
    gt_annotation_frames,
    pred_annotation_frames,
    classes,
    use_superclass=True,
    iou_thresholds=None,
    num_pr_points=50,
    difficulty_mode="Overall&Distance",
    ap_with_heading=True,
    num_parts=100,
    print_ok=False,
    prediction_type=None,
):
    if iou_thresholds is None:
        if use_superclass:
            iou_thresholds = superclass_iou_threshold_dict
        else:
            iou_thresholds = iou_threshold_dict

    assert len(gt_annotation_frames) == len(pred_annotation_frames), "the number of GT must match predictions"
    assert difficulty_mode in ["EASY", "MODERATE", "HARD", "OVERALL"], "difficulty mode is not supported"

    if use_superclass:
        if (
            ("CAR" in classes)
            or ("BUS" in classes)
            or ("TRUCK" in classes)
            or ("TRAILER" in classes)
            or ("VAN" in classes)
            or ("OTHER" in classes)
        ):
            assert (
                ("CAR" in classes) and ("BUS" in classes) and ("TRUCK" in classes)
            ), "CAR/BUS/TRUCK must all exist for vehicle detection"

        # check if labels are present for each class
        num_vehicles = 0
        num_pedestrians = 0
        num_bicycles = 0
        for gt_anno_frame in gt_annotation_frames:
            for object_class in gt_anno_frame["name"]:
                if object_class.upper() in ["CAR", "TRUCK", "BUS", "TRAILER", "VAN", "EMERGENCY_VEHICLE", "OTHER"]:
                    num_vehicles += 1
                if object_class.upper() == "PEDESTRIAN":
                    num_pedestrians += 1
                if object_class.upper() in ["BICYCLE", "MOTORCYCLE"]:
                    num_bicycles += 1

        classes = []
        if num_vehicles > 0:
            classes.append("VEHICLE")
        if num_pedestrians > 0:
            classes.append("PEDESTRIAN")
        if num_bicycles > 0:
            classes.append("BICYCLE")

    num_samples = len(gt_annotation_frames)
    split_parts = compute_split_parts(num_samples, num_parts)
    ious = compute_iou3d_cpu(gt_annotation_frames, pred_annotation_frames, prediction_type=prediction_type)
    num_classes = len(classes)
    num_difficulties = 4
    difficulty_types = ["overall_0_inf", "0-40m", "40-50m", "50m-64"]
    precision = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    recall = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    iou_3d = np.zeros([num_classes, num_difficulties])
    pos_err = np.zeros([num_classes, num_difficulties])
    rot_err = np.zeros([num_classes, num_difficulties])

    gt_class_occurrence = {}
    pred_class_occurrence = {}
    for cur_class in classes:
        gt_class_occurrence[cur_class] = 0
        pred_class_occurrence[cur_class] = 0

    for sample_idx in range(num_samples):
        gt_anno = gt_annotation_frames[sample_idx]
        pred_anno = pred_annotation_frames[sample_idx]

        if len(gt_anno["name"]) == 0 or len(pred_anno["name"]) == 0:
            print("no gt or prediction")
            continue

        if use_superclass:
            if gt_anno["name"].size > 0:
                n_pedestrians = (gt_anno["name"] == "PEDESTRIAN").sum()
                n_bicylces = (np.logical_or(gt_anno["name"] == "BICYCLE", gt_anno["name"] == "MOTORCYCLE")).sum()
                n_vehicles = len(gt_anno["name"]) - n_pedestrians - n_bicylces

                for obj_class in classes:
                    if obj_class.upper() == "VEHICLE":
                        gt_class_occurrence[obj_class] += n_vehicles
                    if obj_class.upper() == "BICYCLE":
                        gt_class_occurrence[obj_class] += n_bicylces
                    if obj_class.upper() == "PEDESTRIAN":
                        gt_class_occurrence[obj_class] += n_pedestrians
            if pred_anno["name"].size > 0:
                n_pedestrians = (pred_anno["name"] == "PEDESTRIAN").sum()
                n_bicylces = (np.logical_or(pred_anno["name"] == "BICYCLE", pred_anno["name"] == "MOTORCYCLE")).sum()
                n_vehicles = len(pred_anno["name"]) - n_pedestrians - n_bicylces
                for obj_class in classes:
                    if obj_class.upper() == "VEHICLE":
                        pred_class_occurrence[obj_class] += n_vehicles
                    if obj_class.upper() == "BICYCLE":
                        pred_class_occurrence[obj_class] += n_bicylces
                    if obj_class.upper() == "PEDESTRIAN":
                        pred_class_occurrence[obj_class] += n_pedestrians
        else:
            for cur_class in classes:
                if gt_anno["name"].size > 0:
                    gt_class_occurrence[cur_class] += (gt_anno["name"] == cur_class.upper()).sum()
                if pred_anno["name"].size > 0:
                    pred_class_occurrence[cur_class] += (pred_anno["name"] == cur_class.upper()).sum()

    for cls_idx, cur_class in enumerate(classes):
        iou_threshold = iou_thresholds[cur_class.upper()]
        for diff_idx in range(num_difficulties):
            ### filter data & determine score thresholds on p-r curve ###
            accum_all_scores, accum_all_ious, accum_all_pos, accum_all_rot, gt_flags, pred_flags = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            num_valid_gt = 0

            for sample_idx in range(num_samples):
                gt_anno = gt_annotation_frames[sample_idx]
                pred_anno = pred_annotation_frames[sample_idx]

                pred_score = pred_anno["score"]
                if len(ious) > 0:
                    iou = ious[sample_idx]
                    gt_flag, pred_flag = filter_data(
                        gt_anno,
                        pred_anno,
                        difficulty_mode,
                        difficulty_level=diff_idx,
                        class_name=cur_class.upper(),
                        use_superclass=use_superclass,
                    )
                    gt_flags.append(gt_flag)
                    pred_flags.append(pred_flag)
                    num_valid_gt += sum(gt_flag == 0)
                    if iou.size > 0:
                        accum_scores, accum_iou, accum_pos, accum_rot = accumulate_scores(
                            gt_anno["boxes_3d"],
                            pred_anno["boxes_3d"],
                            iou,
                            pred_score,
                            gt_flag,
                            pred_flag,
                            iou_threshold=iou_threshold,
                        )
                    else:
                        # continue
                        print("iou is empty")
                        accum_scores, accum_iou, accum_pos, accum_rot = (
                            np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([]),
                        )
                    accum_all_scores.append(accum_scores)
                    accum_all_ious.append(accum_iou)
                    accum_all_pos.append(accum_pos)
                    accum_all_rot.append(accum_rot)
                else:
                    print("No iou found in data. Use an iou threshold of e.g. iou=0.7")

            all_scores = np.concatenate(accum_all_scores, axis=0)
            all_ious = np.concatenate(accum_all_ious, axis=0)
            all_pos = np.concatenate(accum_all_pos, axis=0)
            all_rot = np.concatenate(accum_all_rot, axis=0)
            thresholds = get_thresholds(all_scores, num_valid_gt, num_pr_points=num_pr_points)

            ### compute avg iou, pos/rot error ###
            iou_3d[cls_idx, diff_idx] = np.average(all_ious) if len(all_ious) else 0.0
            pos_err[cls_idx, diff_idx] = np.average(all_pos) if len(all_pos) else 0.0
            rot_err[cls_idx, diff_idx] = np.average(all_rot) if len(all_rot) else 0.0

            ### compute tp/fp/fn ###
            confusion_matrix = np.zeros([len(thresholds), 3])  # only record tp/fp/fn
            for sample_idx in range(num_samples):
                pred_score = pred_annotation_frames[sample_idx]["score"]
                iou = ious[sample_idx]
                gt_flag, pred_flag = gt_flags[sample_idx], pred_flags[sample_idx]
                for th_idx, score_th in enumerate(thresholds):
                    if iou.size > 0:
                        tp, fp, fn = compute_statistics(
                            iou, pred_score, gt_flag, pred_flag, score_threshold=score_th, iou_threshold=iou_threshold
                        )
                        confusion_matrix[th_idx, 0] += tp
                        confusion_matrix[th_idx, 1] += fp
                        confusion_matrix[th_idx, 2] += fn

            ### draw p-r curve ###
            for th_idx in range(len(thresholds)):
                recall[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / (
                    confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 2]
                )
                precision[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / (
                    confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 1]
                )

            for th_idx in range(len(thresholds)):
                precision[cls_idx, diff_idx, th_idx] = np.max(precision[cls_idx, diff_idx, th_idx:], axis=-1)
                recall[cls_idx, diff_idx, th_idx] = np.max(recall[cls_idx, diff_idx, th_idx:], axis=-1)

    AP = 0

    for i in range(1, precision.shape[-1]):
        AP += precision[..., i]
    AP = AP / num_pr_points * 100

    ret_str = "|%-18s|" % "Classes"
    ret_str += "%-12s|" % "Precision"
    ret_str += "%-12s|" % "Recall"
    ret_str += "\n"
    for idx, cur_class in enumerate(classes):
        ret_str += "|%-18s|" % cur_class
        ret_str += "%-12.2f|" % (np.mean(precision[idx], axis=-1)[0] * 100)
        ret_str += "%-12.2f|" % (np.mean(recall[idx], axis=-1)[0] * 100)
        ret_str += "\n"
    print(ret_str)
    ret_dict = {}

    ret_str = "|AP@%-15s|" % (str(num_pr_points))
    for diff_type in difficulty_types:
        ret_str += "%-15s|" % diff_type
    ret_str += "%-20s|" % "Occurrence (pred/gt)"
    ret_str += "%-10s|" % "IOU_3D"
    ret_str += "%-10s|" % "Pos-RMSE"
    ret_str += "%-10s|" % "Rot-RMSE"
    ret_str += "\n"
    for cls_idx, cur_class in enumerate(classes):
        ret_str += "|%-18s|" % cur_class
        for diff_idx in range(num_difficulties):
            diff_type = difficulty_types[diff_idx]
            key = "AP_" + cur_class + "/" + diff_type
            # TODO: Adopt correction of TP=0, FP=0 -> AP = 0 for all difficulty
            # types by counting occurrence individually for each difficulty type
            # if pred_class_occurrence[cur_class] == 0 and gt_class_occurrence[cur_class] == 0:
            #     AP[cls_idx, diff_idx] = 100
            ap_score = AP[cls_idx, diff_idx]
            ret_dict[key] = ap_score
            ret_str += "%-15.2f|" % ap_score
        ret_str += "%-20s|" % (str(pred_class_occurrence[cur_class]) + "/" + str(gt_class_occurrence[cur_class]))
        ret_str += "%-10.2f|" % np.average(iou_3d[cls_idx].flatten())
        ret_str += "%-10.2f|" % np.average(pos_err[cls_idx].flatten())
        ret_str += "%-10.2f|" % np.average(rot_err[cls_idx].flatten())
        ret_str += "\n"
    mAP = np.mean(AP, axis=0)
    ret_str += "|%-18s|" % "mAP"
    for diff_idx in range(num_difficulties):
        diff_type = difficulty_types[diff_idx]
        key = "AP_mean" + "/" + diff_type
        ap_score = mAP[diff_idx]
        ret_dict[key] = ap_score
        ret_str += "%-15.2f|" % ap_score
    ret_str += "%-20s|" % (
        str(np.sum(list(pred_class_occurrence.values())))
        + "/"
        + str(np.sum(list(gt_class_occurrence.values())))
        + " (Total)"
    )
    ret_str += "%-10.2f|" % np.average(iou_3d.flatten())
    ret_str += "%-10.2f|" % np.average(pos_err.flatten())
    ret_str += "%-10.2f|" % np.average(rot_err.flatten())
    ret_str += "\n"

    if print_ok:
        print(ret_str)

    ####################
    ## pretty print (for excel sheet)
    ####################
    ret_header_str = "Class,Precision,Recall,AP_overall,distance_0_40,distance_40_50,distance_50_64,Occurrence (pred/gt),IOU_3D,Pos-RMSE,Rot-RMSE\n"
    for cls_idx, cur_class in enumerate(classes):
        ret_header_str += f"{cur_class},{np.mean(precision[cls_idx], axis=-1)[0] * 100:.2f},{np.mean(recall[cls_idx], axis=-1)[0] * 100:.2f},"
        for diff_idx in range(num_difficulties):
            diff_type = difficulty_types[diff_idx]
            ap_score = AP[cls_idx, diff_idx]
            ret_header_str += f"{ap_score:.2f},"
        ret_header_str += f"{pred_class_occurrence[cur_class]}/{gt_class_occurrence[cur_class]},"
        ret_header_str += f"{np.average(iou_3d[cls_idx].flatten()):.2f},"
        ret_header_str += f"{np.average(pos_err[cls_idx].flatten()):.2f},"
        ret_header_str += f"{np.average(rot_err[cls_idx].flatten()):.2f}\n"

    ret_header_str += "mAP,,,"
    for diff_idx in range(num_difficulties):
        diff_type = difficulty_types[diff_idx]
        ap_score = mAP[diff_idx]
        ret_header_str += f"{ap_score:.2f},"
    ret_header_str += f"{np.sum(list(pred_class_occurrence.values()))}/{np.sum(list(gt_class_occurrence.values()))},"
    ret_header_str += f"{np.average(iou_3d.flatten()):.2f},"
    ret_header_str += f"{np.average(pos_err.flatten()):.2f},"
    ret_header_str += f"{np.average(rot_err.flatten()):.2f}\n"

    # print pretty table results for excel sheet
    # print(ret_header_str)
    ####################
    return ret_str, ret_dict


@numba.jit(nopython=True)
def get_thresholds(scores, num_gt, num_pr_points):
    eps = 1e-6
    scores.sort()
    scores = scores[::-1]
    recall_level = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (r_recall + l_recall < 2 * recall_level) and i < (len(scores) - 1):
            continue
        thresholds.append(score)
        recall_level += 1 / num_pr_points

        while r_recall + l_recall + eps > 2 * recall_level:
            thresholds.append(score)
            recall_level += 1 / num_pr_points
    return thresholds


# TODO: only use annotation in live mode (comment for debugging)
@numba.jit(nopython=True)
def accumulate_scores(gt_shapes, pred_shapes, iou, pred_scores, gt_flag, pred_flag, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assert num_gt == len(gt_shapes)
    assert num_pred == len(pred_shapes)

    assigned = np.full(num_pred, False)
    accum_scores = np.zeros(num_gt)
    accum_ious = np.zeros(num_gt)
    accum_pos_rmse = np.zeros(num_gt)
    accum_rot_rmse = np.zeros(num_gt)
    accum_idx = 0
    for gt_id, gt_shape in enumerate(gt_shapes):
        if gt_flag[gt_id] == -1:  # not the same class
            continue
        det_idx = -1
        detected_score = -1
        det_iou = -1
        det_pos_rmse = -1
        det_rot_rmse = -1
        for pred_id, pred_shape in enumerate(pred_shapes):
            if pred_flag[pred_id] == -1:  # not the same class
                continue
            if assigned[pred_id]:
                continue
            iou_ij = iou[gt_id, pred_id]
            pred_score = pred_scores[pred_id]
            if (iou_ij > iou_threshold) and (pred_score > detected_score) and (iou_ij > det_iou):
                det_idx = pred_id
                detected_score = pred_score
                det_iou = iou_ij
                det_pos_rmse = np.linalg.norm(gt_shape[:3] - pred_shape[:3])
                det_rot_rmse = abs(gt_shape[6] - pred_shape[6]) % math.pi
                if det_rot_rmse > math.pi * 0.5:
                    det_rot_rmse = det_rot_rmse - math.pi * 0.5

        if (detected_score == -1) and (gt_flag[gt_id] == 0):  # false negative
            pass
        elif (detected_score != -1) and (gt_flag[gt_id] == 1 or pred_flag[det_idx] == 1):  # ignore
            assigned[det_idx] = True
        elif detected_score != -1:  # true positive
            accum_scores[accum_idx] = pred_scores[det_idx]
            accum_ious[accum_idx] = det_iou
            accum_pos_rmse[accum_idx] = det_pos_rmse
            accum_rot_rmse[accum_idx] = det_rot_rmse
            accum_idx += 1
            assigned[det_idx] = True

    return accum_scores[:accum_idx], accum_ious[:accum_idx], accum_pos_rmse[:accum_idx], accum_rot_rmse[:accum_idx]


@numba.jit(nopython=True)
def compute_statistics(iou, pred_scores, gt_flag, pred_flag, score_threshold, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    under_threshold = pred_scores < score_threshold

    tp, fp, fn = 0, 0, 0
    for i in range(num_gt):
        if gt_flag[i] == -1:  # different classes
            continue
        det_idx = -1
        detected = False
        best_matched_iou = 0
        gt_assigned_to_ignore = False

        for j in range(num_pred):
            if pred_flag[j] == -1:  # different classes
                continue
            if assigned[j]:  # already assigned to other GT
                continue
            if under_threshold[j]:  # compute only boxes above threshold
                continue
            iou_ij = iou[i, j]
            if (iou_ij > iou_threshold) and (iou_ij > best_matched_iou or gt_assigned_to_ignore) and pred_flag[j] == 0:
                best_matched_iou = iou_ij
                det_idx = j
                detected = True
                gt_assigned_to_ignore = False
            elif (iou_ij > iou_threshold) and (not detected) and pred_flag[j] == 1:
                det_idx = j
                detected = True
                gt_assigned_to_ignore = True

        if (not detected) and gt_flag[i] == 0:  # false negative
            fn += 1
        elif detected and (gt_flag[i] == 1 or pred_flag[det_idx] == 1):  # ignore
            assigned[det_idx] = True
        elif detected:  # true positive
            tp += 1
            assigned[det_idx] = True

    for j in range(num_pred):
        if not (assigned[j] or pred_flag[j] == -1 or pred_flag[j] == 1 or under_threshold[j]):
            fp += 1

    return tp, fp, fn


def filter_data(gt_anno, pred_anno, difficulty_mode, difficulty_level, class_name, use_superclass):
    """
    Filter data by class name and difficulty

    Args:
        gt_anno:
        pred_anno:
        difficulty_mode:
        difficulty_level:
        class_name:

    Returns:
        gt_flags/pred_flags:
            1 : same class but ignored with different difficulty levels
            0 : accepted
           -1 : rejected with different classes
    """
    num_gt = len(gt_anno["name"])
    gt_flag = np.zeros(num_gt, dtype=np.int64)
    if num_gt > 0:
        if use_superclass:
            if class_name == "VEHICLE":
                reject = np.logical_or(
                    gt_anno["name"] == "PEDESTRIAN",
                    np.logical_or(gt_anno["name"] == "BICYCLE", gt_anno["name"] == "MOTORCYCLE"),
                )
            elif class_name == "BICYCLE":
                reject = ~np.logical_or(gt_anno["name"] == "BICYCLE", gt_anno["name"] == "MOTORCYCLE")
            else:
                reject = gt_anno["name"] != class_name
        else:
            reject = gt_anno["name"] != class_name
        gt_flag[reject] = -1
    num_pred = len(pred_anno["name"])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if num_pred > 0:
        if use_superclass:
            if class_name == "VEHICLE":
                reject = np.logical_or(
                    pred_anno["name"] == "PEDESTRIAN",
                    np.logical_or(pred_anno["name"] == "BICYCLE", pred_anno["name"] == "MOTORCYCLE"),
                )
            elif class_name == "BICYCLE":
                reject = ~np.logical_or(pred_anno["name"] == "BICYCLE", pred_anno["name"] == "MOTORCYCLE")
            else:
                reject = pred_anno["name"] != class_name
        else:
            reject = pred_anno["name"] != class_name
        pred_flag[reject] = -1

    if difficulty_mode == "OVERALL":
        ignore = overall_filter(gt_anno["boxes_3d"], difficulty_level)
        gt_flag[ignore] = 1
        ignore = overall_filter(pred_anno["boxes_3d"], difficulty_level)
        pred_flag[ignore] = 1

    return gt_flag, pred_flag


def iou3d_kernel(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    iou3d = intersection_3d / union_3d
    return iou3d


def iou3d_kernel_with_heading(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    iou3d = intersection_3d / union_3d

    # rotation orientation filtering
    diff_rot = gt_boxes[:, [6]] - pred_boxes[:, [6]].T
    diff_rot = np.abs(diff_rot)
    reverse_diff_rot = 2 * np.pi - diff_rot
    diff_rot[diff_rot >= np.pi] = reverse_diff_rot[diff_rot >= np.pi]  # constrain to [0-pi]
    iou3d[diff_rot > np.pi / 2] = 0  # unmatched if diff_rot > 90
    return iou3d


def rotate_iou_kernel_eval(gt_boxes, pred_boxes):
    iou3d_cpu = rotate_iou_cpu_eval(gt_boxes, pred_boxes)
    return iou3d_cpu


def compute_iou3d(gt_annos, pred_annos, split_parts, with_heading):
    """
    Compute iou3d of all samples by parts

    Args:
        with_heading: filter with heading
        gt_annos: list of dicts for each sample
        pred_annos:
        split_parts: for part-based iou computation

    Returns:
        ious: list of iou arrays for each sample
    """
    gt_num_per_sample = np.stack([len(anno["name"]) for anno in gt_annos], 0)
    pred_num_per_sample = np.stack([len(anno["name"]) for anno in pred_annos], 0)
    ious = []
    sample_idx = 0
    for num_part_samples in split_parts:
        gt_annos_part = gt_annos[sample_idx : sample_idx + num_part_samples]
        pred_annos_part = pred_annos[sample_idx : sample_idx + num_part_samples]

        gt_boxes = np.concatenate([anno["boxes_3d"] for anno in gt_annos_part], 0)
        pred_boxes = np.concatenate([anno["boxes_3d"] for anno in pred_annos_part], 0)

        if with_heading:
            iou3d_part = iou3d_kernel_with_heading(gt_boxes, pred_boxes)
        else:
            iou3d_part = iou3d_kernel(gt_boxes, pred_boxes)

        gt_num_idx, pred_num_idx = 0, 0
        for idx in range(num_part_samples):
            gt_box_num = gt_num_per_sample[sample_idx + idx]
            pred_box_num = pred_num_per_sample[sample_idx + idx]
            ious.append(iou3d_part[gt_num_idx : gt_num_idx + gt_box_num, pred_num_idx : pred_num_idx + pred_box_num])
            gt_num_idx += gt_box_num
            pred_num_idx += pred_box_num
        sample_idx += num_part_samples
    return ious


def compute_iou3d_cpu(gt_annos, pred_annos, prediction_type=None):
    ious = []
    gt_num = len(gt_annos)
    for i in range(gt_num):
        gt_boxes = gt_annos[i]["boxes_3d"]
        pred_boxes = pred_annos[i]["boxes_3d"]

        iou3d_part = rotate_iou_cpu_eval(gt_boxes, pred_boxes)
        ious.append(iou3d_part)
    return ious


def get_attribute_by_name(attribute_list, attribute_name):
    for attribute in attribute_list:
        if attribute["name"] == attribute_name:
            return attribute
    return None


def load_lidar_boxes_into_s110(input_folder_path, object_min_points=0, ouster_lidar_only=False, prediction_type=None):
    def append_object(num_points, l, w, h, rotation, position_3d, category, prediction_type):
        # Ground truth is labeled with camera data, so there are objects
        # contained in the ground truth without a single corresponding
        # point in the LiDAR point cloud.
        # You can specify how many minimum points there should be before a label
        # is included.
        if num_points >= object_min_points:
            name.append(category.upper())
            boxes_3d.append(np.hstack((position_3d, l, w, h, rotation)))
            num_points_in_gt.append(num_points)

    labels_file_paths = listdir_fullpath(input_folder_path)
    labels_list = []

    for label_file_path in labels_file_paths:
        if ouster_lidar_only and "ouster" not in label_file_path:
            continue
        name = []
        boxes_3d = []
        num_points_in_gt = []
        json_data = json.load(open(label_file_path))
        scores = []
        if "openlabel" in json_data:
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                if len(frame_obj["objects"].items()) == 0:
                    print("no detections in frame: {}".format(label_file_path))
                    continue
                for object_id, label in frame_obj["objects"].items():
                    # Dataset in ASAM OpenLABEL format
                    l = float(label["object_data"]["cuboid"]["val"][7])
                    w = float(label["object_data"]["cuboid"]["val"][8])
                    h = float(label["object_data"]["cuboid"]["val"][9])
                    quat_x = float(label["object_data"]["cuboid"]["val"][3])
                    quat_y = float(label["object_data"]["cuboid"]["val"][4])
                    quat_z = float(label["object_data"]["cuboid"]["val"][5])
                    quat_w = float(label["object_data"]["cuboid"]["val"][6])
                    if np.linalg.norm([quat_x, quat_y, quat_z, quat_w]) == 0.0:
                        continue
                    rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w]).apply([l / 2.0, 0, 0])
                    position_3d = [
                        float(label["object_data"]["cuboid"]["val"][0]),
                        float(label["object_data"]["cuboid"]["val"][1]),
                        float(label["object_data"]["cuboid"]["val"][2]),  # - h / 2  # To avoid floating bounding boxes
                    ]
                    image_pos = perspective.project_from_lidar_south_to_image(
                        np.array(
                            [
                                [position_3d[0], position_3d[0] + rotation[0]],
                                [position_3d[1], position_3d[1] + rotation[1]],
                                [position_3d[2] - h / 2, position_3d[2] - h / 2 + rotation[2]],
                            ]
                        )
                    )

                    # 3d position in s110_base with z=0
                    position_3d_in_s110_base = perspective.project_to_ground(image_pos)
                    orientation_vec = position_3d_in_s110_base[:, 0] - position_3d_in_s110_base[:, 1]
                    # yaw rotation in s110_base frame
                    rotation_yaw = np.arctan2(orientation_vec[1], orientation_vec[0])

                    attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "num_points")

                    num_points = 0
                    if attribute is not None:
                        num_points = int(float(attribute["val"]))
                    append_object(
                        num_points,
                        l,
                        w,
                        h,
                        rotation_yaw,
                        position_3d_in_s110_base[:, 0],
                        label["object_data"]["type"],
                        prediction_type=prediction_type,
                    )

                    attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "score")
                    if attribute is not None:
                        score = attribute["val"]
                        scores.append(score)
        else:
            for label in json_data["labels"]:
                if "dimensions" in label:
                    # Dataset R1 NOT IN ASAM OpenLABEL format
                    l = float(label["dimensions"]["length"])
                    w = float(label["dimensions"]["width"])
                    h = float(label["dimensions"]["height"])
                    quat_x = float(label["rotation"]["_x"])
                    quat_y = float(label["rotation"]["_y"])
                    quat_z = float(label["rotation"]["_z"])
                    quat_w = float(label["rotation"]["_w"])
                    rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("zyx", degrees=False)[0]
                    position_3d = [
                        float(label["center"]["x"]),
                        float(label["center"]["y"]),
                        float(label["center"]["z"]) - h / 2,  # To avoid floating bounding boxes
                    ]
                else:
                    # Dataset R0 NOT IN ASAM OpenLABEL format
                    l = float(label["box3d"]["dimension"]["length"])
                    w = float(label["box3d"]["dimension"]["width"])
                    h = float(label["box3d"]["dimension"]["height"])
                    rotation = float(label["box3d"]["orientation"]["rotationYaw"])
                    position_3d = [
                        float(label["box3d"]["location"]["x"]),
                        float(label["box3d"]["location"]["y"]),
                        float(label["box3d"]["location"]["z"]),
                    ]
                num_points = 0
                append_object(num_points, l, w, h, rotation, position_3d, label["category"])

        label_dict = {
            "name": np.array(name),
            "boxes_3d": np.array(boxes_3d),
            "num_points_in_gt": np.array(num_points_in_gt),
            "score": np.array(scores),
        }
        labels_list.append(label_dict)
    return labels_list


def parse_input_files(input_folder_path, object_min_points=0, ouster_lidar_only=False, prediction_type=None):
    def append_object(num_points, l, w, h, rotation, position_3d, category, prediction_type):
        # Ground truth is labeled with camera data, so there are objects
        # contained in the ground truth without a single corresponding
        # point in the LiDAR point cloud.
        # You can specify how many minimum points there should be before a label
        # is included.
        name.append(category.upper())
        boxes_3d.append(np.hstack((position_3d, l, w, h, rotation)))
        num_points_in_gt.append(num_points)

    labels_file_list = listdir_fullpath(input_folder_path)
    labels_file_list.sort()
    labels_list = []

    for label_file in labels_file_list:
        if ouster_lidar_only and "ouster" not in label_file:
            continue
        name = []
        boxes_3d = []
        num_points_in_gt = []
        json_file = open(
            label_file,
        )
        json_data = json.load(json_file)
        scores = []
        if "openlabel" in json_data:
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                for object_id, label in frame_obj["objects"].items():

                    # Dataset in ASAM OpenLABEL format
                    l = float(label["object_data"]["cuboid"]["val"][7])
                    w = float(label["object_data"]["cuboid"]["val"][8])
                    h = float(label["object_data"]["cuboid"]["val"][9])
                    quat_x = float(label["object_data"]["cuboid"]["val"][3])
                    quat_y = float(label["object_data"]["cuboid"]["val"][4])
                    quat_z = float(label["object_data"]["cuboid"]["val"][5])
                    quat_w = float(label["object_data"]["cuboid"]["val"][6])
                    if np.linalg.norm([quat_x, quat_y, quat_z, quat_w]) == 0.0:
                        continue
                    rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("zyx", degrees=False)[0]
                    position_3d = [
                        float(label["object_data"]["cuboid"]["val"][0]),
                        float(label["object_data"]["cuboid"]["val"][1]),
                        float(label["object_data"]["cuboid"]["val"][2]),  # - h / 2  # To avoid floating bounding boxes
                    ]
                    attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "num_points")

                    num_points = 10  # Dummy value.
                    if (prediction_type is None) or (attribute is not None and "lidar3d" in prediction_type):
                        num_points = int(attribute["val"])

                    append_object(
                        num_points,
                        l,
                        w,
                        h,
                        rotation,
                        position_3d,
                        label["object_data"]["type"],
                        prediction_type=prediction_type,
                    )

                    attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "score")
                    if attribute is not None:
                        score = attribute["val"]
                        scores.append(score)
        else:
            for label in json_data["labels"]:
                if "dimensions" in label:
                    # Dataset R1 NOT IN ASAM OpenLABEL format
                    l = float(label["dimensions"]["length"])
                    w = float(label["dimensions"]["width"])
                    h = float(label["dimensions"]["height"])
                    quat_x = float(label["rotation"]["_x"])
                    quat_y = float(label["rotation"]["_y"])
                    quat_z = float(label["rotation"]["_z"])
                    quat_w = float(label["rotation"]["_w"])
                    rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("zyx", degrees=False)[0]
                    position_3d = [
                        float(label["center"]["x"]),
                        float(label["center"]["y"]),
                        float(label["center"]["z"]) - h / 2,  # To avoid floating bounding boxes
                    ]
                else:
                    # Dataset R0 NOT IN ASAM OpenLABEL format
                    l = float(label["box3d"]["dimension"]["length"])
                    w = float(label["box3d"]["dimension"]["width"])
                    h = float(label["box3d"]["dimension"]["height"])
                    rotation = float(label["box3d"]["orientation"]["rotationYaw"])
                    position_3d = [
                        float(label["box3d"]["location"]["x"]),
                        float(label["box3d"]["location"]["y"]),
                        float(label["box3d"]["location"]["z"]),
                    ]
                num_points = 0
                append_object(num_points, l, w, h, rotation, position_3d, label["category"])

        json_file.close()
        label_dict = {
            "name": np.array(name),
            "boxes_3d": np.array(boxes_3d),
            "num_points_in_gt": np.array(num_points_in_gt),
            "score": np.array(scores),
        }
        labels_list.append(label_dict)
    return labels_list


def get_pc_path(labels_path):
    parent_dir = os.path.dirname(labels_path)
    subdirs = os.listdir(parent_dir)
    label_dir = os.path.basename(labels_path)
    if subdirs[0] == label_dir:
        return os.path.join(parent_dir, subdirs[1])
    else:
        return os.path.join(parent_dir, subdirs[0])


def listdir_fullpath(d):
    # add all file paths into list using glob
    return sorted(glob.glob(os.path.join(d, "*.json")))


def prepare_predictions_kitti(predictions_path):
    if os.path.isfile(predictions_path):
        predictions_file_list = [predictions_path]
    else:
        predictions_file_list = sorted(glob.glob(os.path.join(predictions_path, "*.txt")))
        predictions_file_list.sort()
    predictions_list = []

    for prediction_file in predictions_file_list:
        name = []
        boxes_3d = []
        with open(prediction_file, "r") as f:
            for line in f:
                line = line.rstrip().split()
                prediction = [float(item) for item in line[1:]]

                position_3d = [
                    prediction[0],
                    prediction[1],
                    prediction[2],
                ]
                yaw_radian = prediction[6]
                rotation = np.array([0, 0, yaw_radian])
                h = prediction[5]
                image_pos = perspective.project_from_lidar_south_to_image(
                    np.array(
                        [
                            [position_3d[0], position_3d[0] + rotation[0]],
                            [position_3d[1], position_3d[1] + rotation[1]],
                            [position_3d[2] - h / 2, position_3d[2] - h / 2 + rotation[2]],
                        ]
                    )
                )

                # 3d position in s110_base with z=0
                position_3d_in_s110_base = perspective.project_to_ground(image_pos)
                orientation_vec = position_3d_in_s110_base[:, 0] - position_3d_in_s110_base[:, 1]
                # yaw rotation in s110_base frame
                rotation_yaw = np.arctan2(orientation_vec[1], orientation_vec[0])
                prediction[0] = position_3d_in_s110_base[0, 0]
                prediction[1] = position_3d_in_s110_base[1, 0]
                prediction[2] = position_3d_in_s110_base[2, 0]
                prediction[6] = rotation_yaw

                prediction.insert(0, line[0])
                name.append(str(prediction[0]).upper())
                boxes_3d.append(prediction[1:])
        prediction_dict = {"name": np.array(name), "boxes_3d": np.array(boxes_3d)}
        predictions_list.append(prediction_dict)
    return predictions_list


def visualize_bounding_boxes(
    label_path, prediction_path, object_min_points=0, ouster_lidar_only=False, prediction_type=None
):
    def rreplace(s, old, new):
        """Reverse replace"""
        return (s[::-1].replace(old[::-1], new[::-1], 1))[::-1]

    # read OpenLABEL labels
    gt_data = parse_input_files(label_path, object_min_points, ouster_lidar_only)
    # FOR KITTI
    # pred_data = prepare_predictions(prediction_path)
    # read OpenLABEL predictions/labels
    pred_data = parse_input_files(prediction_path, object_min_points, ouster_lidar_only)

    for item in pred_data:
        n_obj = item["name"].size
        item["score"] = np.full(n_obj, 1)
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
    result_str, _ = get_evaluation_results(
        gt_data, pred_data, classes, use_superclass=False, difficulty_mode="Overall", prediction_type=prediction_type
    )
    print(result_str)

    pc_file = os.path.join(
        get_pc_path(os.path.dirname(label_path)), rreplace(os.path.basename(label_path), ".json", ".pcd")
    )
    pc = o3d.io.read_point_cloud(pc_file)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer")
    vis.get_render_option().background_color = [0.1, 0.1, 0.1]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    vis.add_geometry(pc)
    for box in gt_data[0]["boxes_3d"]:
        obb = o3d.geometry.OrientedBoundingBox(
            box[:3],
            np.array([[np.cos(box[6]), np.sin(box[6]), 0], [-np.sin(box[6]), np.cos(box[6]), 0], [0, 0, 1]]),
            box[3:6],
        )
        obb.color = np.array([0, 1, 0])
        vis.add_geometry(obb)
    for box in pred_data[0]["boxes_3d"]:
        obb = o3d.geometry.OrientedBoundingBox(
            box[:3],
            np.array([[np.cos(box[6]), np.sin(box[6]), 0], [-np.sin(box[6]), np.cos(box[6]), 0], [0, 0, 1]]),
            box[3:6],
        )
        obb.color = np.array([1, 0, 0])
        vis.add_geometry(obb)
    vis.get_view_control().set_zoom(0.05)
    vis.get_view_control().set_front([-0.940, 0.096, 0.327])
    vis.get_view_control().set_lookat([17.053, 0.544, -2.165])
    vis.get_view_control().set_up([0.327, -0.014, 0.945])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    args = parse_arguments()
    camera_id = args.camera_id
    assert camera_id is not None, "Please provide the camera ID you want to use for evaluation."
    folder_path_ground_truth = args.folder_path_ground_truth
    folder_path_predictions = args.folder_path_predictions
    object_min_points = args.object_min_points
    use_superclasses = args.use_superclasses
    file_path_calibration_data = args.file_path_calibration_data
    # possible values: [lidar3d_unsupervised, lidar3d_supervised, mono3d, multi3d]
    prediction_type = args.prediction_type
    prediction_format = args.prediction_format  # possible values: [openlabel, kitti]
    use_ouster_lidar_only = args.use_ouster_lidar_only
    perspective = parse_perspective(file_path_calibration_data)
    if hasattr(perspective, "initialize_matrices"):
        perspective.initialize_matrices()

    if os.path.isfile(folder_path_ground_truth) and os.path.isfile(folder_path_predictions):
        visualize_bounding_boxes(
            folder_path_ground_truth,
            folder_path_predictions,
            object_min_points=object_min_points,
            ouster_lidar_only=use_ouster_lidar_only,
        )
    else:
        gt_data = load_lidar_boxes_into_s110(
            folder_path_ground_truth,
            object_min_points=object_min_points,
            ouster_lidar_only=use_ouster_lidar_only,
            prediction_type=prediction_type,
        )

        if prediction_format == "openlabel":
            # load lidar labels into s110 for all prediction types (mono3d, lidar3d, multi3d)
            # parse openlabel predictions
            if prediction_type in ["lidar3d_supervised", "lidar3d_unsupervised"]:
                pred_data = load_lidar_boxes_into_s110(
                    folder_path_predictions,
                    object_min_points=object_min_points,
                    ouster_lidar_only=use_ouster_lidar_only,
                    prediction_type=prediction_type,
                )
            else:
                pred_data = parse_input_files(
                    folder_path_predictions,
                    object_min_points=object_min_points,
                    ouster_lidar_only=use_ouster_lidar_only,
                    prediction_type=prediction_type,
                )

        elif prediction_format == "kitti":
            # parse KITTI predictions
            pred_data = prepare_predictions_kitti(folder_path_predictions)
        else:
            raise ValueError("Unknown prediction format: {}".format(prediction_format))

        if prediction_type == "lidar3d_unsupervised":
            # set lidar detection scores to 1 (because of unsupervised learning)
            for item in pred_data:
                n_obj = item["name"].size
                item["score"] = np.full(n_obj, 1)

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
        result_str, result_dict = get_evaluation_results(
            gt_data,
            pred_data,
            classes,
            use_superclass=use_superclasses,
            difficulty_mode="OVERALL",
            prediction_type=prediction_type,
        )
        print(result_str)
