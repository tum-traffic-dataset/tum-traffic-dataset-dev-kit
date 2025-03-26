import glob
import json
import math
import os
import numpy as np
import numba
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial import ConvexHull
import time
import argparse
##################################
# Evaluation Script for 3D Object Detection
##################################


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

def get_3d_box(box_size, heading_angle, center):
    """Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    """

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d



def poly_area(x, y):
    """Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

@numba.jit(nopython=True)
def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList

def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_iou(corners1, corners2):
    """Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    """
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]

    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])

    inter_vol = inter_area * max(0.0, ymax - ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d



def get_box3d_iou(box_info):
    gt_box, pred_box = box_info
    if np.linalg.norm(gt_box[:3] - pred_box[:3]) > 5:
        return 0.0

    corners_3d_ground = get_3d_box(gt_box[3:6], gt_box[-1], gt_box[[0, 2, 1]])
    corners_3d_predict = get_3d_box(pred_box[3:6], pred_box[-1], pred_box[[0, 2, 1]])

    iou_3d, _ = box3d_iou(corners_3d_ground, corners_3d_predict)
    return iou_3d


def rotate_iou_cpu_eval(gt_boxes, pred_boxes):
    """
    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes:
    """
    data_list = []
    gt_num = len(gt_boxes)
    for gt_box in gt_boxes:
        for pred_box in pred_boxes:
            data_list.append((gt_box, pred_box))
    with Pool(8) as pool:
        result = pool.map(get_box3d_iou, data_list)
    # For Debugging: same result but no multithread
    # result = []
    # for item in data_list:
    #     result.append(rotate_iou_cpu_one(item))
    result = np.array(result)
    if result.size > 0:
        result = result.reshape((gt_num, -1))
    return result


@numba.jit(nopython=True)
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


def get_evaluation_results(
        gt_annotation_frames,
        pred_annotation_frames,
        classes,
        iou_thresholds=None,
        num_pr_points=50,
        difficulty_mode="OVERALL",
        ap_with_heading=True,
        num_parts=100,
):
    if iou_thresholds is None:
        iou_thresholds = iou_threshold_dict

    assert difficulty_mode in ["EASY", "MODERATE", "HARD", "OVERALL"], "difficulty mode is not supported"

    num_samples = len(gt_annotation_frames)
    split_parts = compute_split_parts(num_samples, num_parts)
    ious = compute_iou3d_cpu(gt_annotation_frames, pred_annotation_frames)
    num_classes = len(classes)
    num_difficulties = 4
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
                        class_name=cur_class.upper()
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

    precision_values = []
    recall_values = []
    for idx, cur_class in enumerate(classes):
        precision_values.append(np.mean(precision[idx], axis=-1)[0] * 100)
        recall_values.append(np.mean(recall[idx], axis=-1)[0] * 100)
    ret_dict = {}
    ret_dict["precision"] = np.mean(precision_values)
    ret_dict["recall"] = np.mean(recall_values)
    ret_dict["3d_map"] = np.mean(AP, axis=0)[0]
    ret_dict["3d_iou"] = np.average(iou_3d.flatten())
    ret_dict["position_rmse"] = np.average(pos_err.flatten())
    ret_dict["rotation_rmse"] = np.average(rot_err.flatten())
    return ret_dict


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


def filter_data(gt_anno, pred_anno, difficulty_mode, difficulty_level, class_name):
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
        reject = gt_anno["name"] != class_name
        gt_flag[reject] = -1
    num_pred = len(pred_anno["name"])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if num_pred > 0:
        reject = pred_anno["name"] != class_name
        pred_flag[reject] = -1

    if difficulty_mode == "OVERALL":
        ignore = overall_filter(gt_anno["boxes_3d"], difficulty_level)
        gt_flag[ignore] = 1
        ignore = overall_filter(pred_anno["boxes_3d"], difficulty_level)
        pred_flag[ignore] = 1

    return gt_flag, pred_flag


def compute_iou3d_cpu(gt_annos, pred_annos):
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

def load_3d_boxes(input_file_path):
    labels_list = []
    name = []
    boxes_3d = []
    num_points_in_gt = []
    json_data = json.load(open(input_file_path))
    scores = []
    if "openlabel" in json_data:
        for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
            if len(frame_obj["objects"].items()) == 0:
                print("no detections in frame: {}".format(input_file_path))
                continue
            for object_id, label in frame_obj["objects"].items():
                # Dataset in ASAM OpenLABEL format
                category = label["object_data"]["type"]
                l = float(label["object_data"]["cuboid"]["val"][7])
                w = float(label["object_data"]["cuboid"]["val"][8])
                h = float(label["object_data"]["cuboid"]["val"][9])
                quat_x = float(label["object_data"]["cuboid"]["val"][3])
                quat_y = float(label["object_data"]["cuboid"]["val"][4])
                quat_z = float(label["object_data"]["cuboid"]["val"][5])
                quat_w = float(label["object_data"]["cuboid"]["val"][6])
                if np.linalg.norm([quat_x, quat_y, quat_z, quat_w]) == 0.0:
                    continue

                # rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler()
                # convert quaternion to euler angle
                rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("zyx")[0]
                position_3d = [
                    float(label["object_data"]["cuboid"]["val"][0]),
                    float(label["object_data"]["cuboid"]["val"][1]),
                    float(label["object_data"]["cuboid"]["val"][2]),  # - h / 2  # To avoid floating bounding boxes
                ]

                attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "num_points")
                num_points = 0
                if attribute is not None:
                    num_points = int(float(attribute["val"]))

                # Specify how many minimum points there should be before a label is included.
                # if num_points >= 5:
                name.append(category.upper())
                boxes_3d.append(np.hstack((position_3d, l, w, h, rotation_yaw)))
                num_points_in_gt.append(num_points)

                attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "score")
                if attribute is not None:
                    score = attribute["val"]
                    scores.append(score)

        label_dict = {
            "name": np.array(name),
            "boxes_3d": np.array(boxes_3d),
            "num_points_in_gt": np.array(num_points_in_gt),
            "score": np.array(scores),
        }
        labels_list.append(label_dict)
    return labels_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for 3D Object Detection")
    parser.add_argument(
        "--test_annotation_file_path",
        type=str,
        required=True,
        help="File path to test annotation file",
    )
    parser.add_argument(
        "--user_submission_file_path",
        type=str,
        required=True,
        help="File path to user submission file",
    )
    args = parser.parse_args()
    test_annotation_file = args.test_annotation_file_path
    user_submission_file = args.user_submission_file_path
    print("Starting Challenge Evaluation.....")
    start_time = time.time()
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
    gt_data = load_3d_boxes(test_annotation_file)
    pred_data = load_3d_boxes(user_submission_file)

    result_dict = get_evaluation_results(
        gt_data,
        pred_data,
        classes,
        difficulty_mode="OVERALL",
    )

    print("Evaluating for Test Phase")
    print("Precision: ", result_dict["precision"])
    print("Recall: ", result_dict["recall"])
    print("3D_IoU: ", result_dict["3d_iou"])
    print("Position_RMSE: ", result_dict["position_rmse"])
    print("Rotation_RMSE: ", result_dict["rotation_rmse"])
    print("3D_mAP: ", result_dict["3d_map"])

    print("Completed evaluation for Test Phase")
    print("Time taken for evaluation in seconds: ", time.time() - start_time)