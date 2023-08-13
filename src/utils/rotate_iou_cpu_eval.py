# 3D IoU caculate code for 3D object detection
# Kent 2018/12

import numpy as np
from multiprocessing import Pool
from scipy.spatial import ConvexHull
from pytorch3d.ops import box3d_overlap
import torch


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


def poly_area(x, y):
    """Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


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


def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


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


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------


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


def rotate_iou_cpu_one(box_info):
    gt_box, pred_box = box_info
    if np.linalg.norm(gt_box[:3] - pred_box[:3]) > 5:
        return 0.0

    # diff_rot = gt_box[6] - pred_box[6]
    # diff_rot = np.abs(diff_rot)
    # if diff_rot > np.pi:
    #     diff_rot = 2 * np.pi - diff_rot
    # if diff_rot > np.pi / 2:
    #     return 0.0

    corners_3d_ground = get_3d_box(gt_box[3:6], gt_box[-1], gt_box[[0, 2, 1]])
    corners_3d_predict = get_3d_box(pred_box[3:6], pred_box[-1], pred_box[[0, 2, 1]])
    # Sanity checks with ground truth revealed instability of this
    # method, so replaced by new method from Facebook
    # iou_3d, _ = box3d_iou(corners_3d_ground, corners_3d_predict)
    # Preparation for Facebook method https://pytorch3d.org/docs/iou3d
    corners_3d_ground = torch.from_numpy(np.expand_dims(corners_3d_ground, axis=0).astype(np.float32))
    corners_3d_predict = torch.from_numpy(np.expand_dims(corners_3d_predict, axis=0).astype(np.float32))
    _, iou_3d_pytorch3d = box3d_overlap(corners_3d_ground, corners_3d_predict)
    iou_3d_pytorch3d = iou_3d_pytorch3d.numpy().item()
    return iou_3d_pytorch3d


def rotate_iou_cpu_eval(gt_boxes, pred_boxes):
    """

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes:

    Returns:

    """
    data_list = []
    gt_num = len(gt_boxes)
    for gt_box in gt_boxes:
        for pred_box in pred_boxes:
            data_list.append((gt_box, pred_box))
    with Pool(8) as pool:
        result = pool.map(rotate_iou_cpu_one, data_list)
    # For Debugging: same result but no multithread
    # result = []
    # for item in data_list:
    #     result.append(rotate_iou_cpu_one(item))
    result = np.array(result)
    if result.size > 0:
        result = result.reshape((gt_num, -1))
    return result
