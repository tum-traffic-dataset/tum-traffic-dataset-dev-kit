import numpy as np
from scipy.spatial.transform import Rotation as R

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

providentia_to_tum_traffic_category_mapping = {
    "PRE_TRACK": "OTHER",
    "OTHER": "OTHER",
    "PEDESTRIAN": "PEDESTRIAN",
    "BIKE": "BICYCLE",
    "CAR": "CAR",
    "TRUCK": "TRUCK",
    "BUS": "BUS",
    "CONSTRUCTION_VEHICLE": "TRUCK",
    "DYNAMIC_TRAFFIC_SIGN": "OTHER",
    "TRAFFICSIGN": "OTHER",
    "ANIMAL": "OTHER",
    "OBSTACLE": "OTHER",
    "CONSTRUCTIONSITEDELIMITER": "OTHER"
}

providentia_id_to_class_name_mapping = ["PRE_TRACK",
                                        "OTHER",
                                        "PEDESTRIAN",
                                        "BIKE",
                                        "CAR",
                                        "TRUCK",
                                        "BUS",
                                        "CONSTRUCTION_VEHICLE",
                                        "DYNAMIC_TRAFFIC_SIGN",
                                        "TRAFFICSIGN",
                                        "ANIMAL",
                                        "OBSTACLE",
                                        "CONSTRUCTIONSITEDELIMITER"]

# TUM Traffic Dataset IDs
id_to_class_name_mapping = {
    "0": {
        "class_label_de": "PKW",
        "class_label_en": "Car",
        "color_hex": "#00ccf6",
        "color_rgb": (0, 204, 246),
        "color_bgr": (246, 204, 0),
        "color_bgr_normalized": (0.964705882, 0.8, 0),
        "color_rgb_normalized": (0, 0.8, 0.96),
    },
    "1": {
        "class_label_de": "LKW",
        "class_label_en": "Truck",
        "color_hex": "#3FE9B9",
        "color_rgb": (63, 233, 185),
        "color_bgr": (185, 233, 63),
        "color_bgr_normalized": (0.71372549, 1, 0.337254902),
        "color_rgb_normalized": (0.25, 0.91, 0.72),
    },
    "2": {
        "class_label_de": "Anhänger",
        "class_label_en": "Trailer",
        "color_hex": "#5AFF7E",
        "color_rgb": (90, 255, 126),
        "color_bgr": (126, 255, 90),
        "color_bgr_normalized": (0.494117647, 1, 0.352941176),
        "color_rgb_normalized": (0.35, 1, 0.49),
    },
    "3": {
        "class_label_de": "Van",
        "class_label_en": "Van",
        "color_hex": "#EBCF36",
        "color_rgb": (235, 207, 54),
        "color_bgr": (54, 207, 235),
        "color_bgr_normalized": (0.211764706, 0.811764706, 0.921568627),
        "color_rgb_normalized": (0.92, 0.81, 0.21),
    },
    "4": {
        "class_label_de": "Motorrad",
        "class_label_en": "Motorcycle",
        "color_hex": "#B9A454",
        "color_rgb": (185, 164, 84),
        "color_bgr": (84, 164, 185),
        "color_bgr_normalized": (0.329411765, 0.643137255, 0.725490196),
        "color_rgb_normalized": (0.72, 0.64, 0.33),
    },
    "5": {
        "class_label_de": "Bus",
        "class_label_en": "Bus",
        "color_hex": "#D98A86",
        "color_rgb": (217, 138, 134),
        "color_bgr": (134, 138, 217),
        "color_bgr_normalized": (0.525490196, 0.541176471, 0.850980392),
        "color_rgb_normalized": (0.85, 0.54, 0.52),
    },
    "6": {
        "class_label_de": "Person",
        "class_label_en": "Pedestrian",
        "color_hex": "#E976F9",
        "color_rgb": (233, 118, 249),
        "color_bgr": (249, 118, 233),
        "color_bgr_normalized": (0.976470588, 0.462745098, 0.91372549),
        "color_rgb_normalized": (0.91, 0.46, 0.97),
    },
    "7": {
        "class_label_de": "Fahrrad",
        "class_label_en": "Bicycle",
        "color_hex": "#B18CFF",
        "color_rgb": (177, 140, 255),
        "color_bgr": (255, 140, 177),
        "color_bgr_normalized": (1, 0.549019608, 0.694117647),
        "color_rgb_normalized": (0.69, 0.55, 1),
    },
    "8": {
        "class_label_de": "Einsatzfahrzeug",
        "class_label_en": "Emergency_Vehicle",
        "color_hex": "#666bfa",
        "color_rgb": (102, 107, 250),
        "color_bgr": (250, 107, 102),
        "color_bgr_normalized": (0.980392157, 0.419607843, 0.4),
        "color_rgb_normalized": (0.4, 0.42, 0.98),
    },
    "9": {
        "class_label_de": "Unbekannt",
        "class_label_en": "Other",
        "color_hex": "#C7C7C7",
        "color_rgb": (199, 199, 199),
        "color_bgr": (199, 199, 199),
        "color_bgr_normalized": (0.780392157, 0.780392157, 0.780392157),
        "color_rgb_normalized": (0.78, 0.78, 0.78),
    },
    "10": {
        "class_label_de": "Nummernschild",
        "class_label_en": "License_Plate",
        "color_hex": "#000000",
        "color_rgb": (0, 0, 0),
        "color_bgr": (0, 0, 0),
        "color_bgr_normalized": (0, 0, 0),
        "color_rgb_normalized": (0, 0, 0),
    },
}

# Mapping object class name to A9 dataset class ID
class_name_to_id_mapping = {
    "CAR": 0,
    "TRUCK": 1,
    "TRAILER": 2,
    "VAN": 3,
    "MOTORCYCLE": 4,
    "BUS": 5,
    "PEDESTRIAN": 6,
    "BICYCLE": 7,
    "EMERGENCY_VEHICLE": 8,
    "OTHER": 9,
    "LICENSE_PLATE_LOCATION": 10,
}

# MS COCO ID to A9 ID mapping
ms_coco_to_a9_mapping = {
    "0": {
        "class_id": 6,  # person ID=0 in MS COCO is mapped to pedestrian ID=6 in A9 dataset
        "color_rgb": (249, 118, 233),
        "class_label_de": "Person",
        "class_label_en": "Pedestrian",
    },
    "1": {
        "class_id": 7,  # bicycle ID=1 in MS COCO is mapped to bike ID=7 in A9 dataset
        "color_rgb": (255, 140, 177),
        "class_label_de": "Fahrrad",
        "class_label_en": "Bicycle",
    },
    "2": {
        "class_id": 0,  # car ID=2 in MS COCO is mapped to car ID=0 in A9 dataset
        "color_rgb": (246, 204, 0),
        "class_label_de": "PKW",
        "class_label_en": "Car",
    },
    "3": {
        "class_id": 4,  # motorcycle ID=3 in MS COCO is mapped to motorcycle ID=4 in A9 dataset
        "color_rgb": (84, 164, 185),
        "class_label_de": "Motorrad",
        "class_label_en": "Motorcycle",
    },
    "5": {
        "class_id": 5,  # bus ID=5 in MS COCO is mapped to bus ID=5 in A9 dataset
        "color_rgb": (134, 138, 217),
        "class_label_de": "Bus",
        "class_label_en": "Bus",
    },
    "7": {
        "class_id": 1,  # truck ID=7 in MS COCO is mapped to truck ID=1 in A9 dataset
        "color_rgb": (182, 255, 86),
        "class_label_de": "LKW",
        "class_label_en": "Truck",
    },
}

# MS COCO class name to MS COCO ID mapping
mscoco_class_name_to_id_mapping = {
    "PEDESTRIAN": 0,
    "BICYCLE": 1,
    "CAR": 2,
    "MOTORCYCLE": 3,
    "BUS": 5,
    "TRUCK": 7,
    "TRAILER": 7,
    "EMERGENCY_VEHICLE": 7,
    "OTHER": 2,
    "VAN": 7,
}


def get_cuboid_corners(cuboid, boxes_coordinate_system_origin=None):
    l = cuboid[7]
    w = cuboid[8]
    h = cuboid[9]
    # Create a bounding box outline
    bounding_box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )
    if boxes_coordinate_system_origin == "common_road":
        # add half of the height to the z coordinate
        bounding_box[2, :] += h / 2

    translation = cuboid[:3]
    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    rotation_quaternion = cuboid[3:7]
    rotation_matrix = R.from_quat(rotation_quaternion).as_matrix()
    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
    return corner_box.transpose()


def check_corners_within_image(corners):
    valid_corners = 0
    for idx in range(len(corners)):
        corner = corners[idx, :]
        if corner[0] >= 0 and corner[0] < IMAGE_WIDTH and corner[1] >= 0 and corner[1] < IMAGE_HEIGHT:
            valid_corners += 1
    if valid_corners > 1:
        return True
    return False


def get_2d_corner_points(cx, cy, length, width, yaw):
    """
    Find the coordinates of the rectangle with given center, length, width and angle of the longer side
    """

    mp1 = [cx + length / 2 * np.cos(yaw), cy + length / 2 * np.sin(yaw)]
    mp3 = [cx - length / 2 * np.cos(yaw), cy - length / 2 * np.sin(yaw)]

    p1 = [mp1[0] - width / 2 * np.sin(yaw), mp1[1] + width / 2 * np.cos(yaw)]
    p2 = [mp3[0] - width / 2 * np.sin(yaw), mp3[1] + width / 2 * np.cos(yaw)]
    p3 = [mp3[0] + width / 2 * np.sin(yaw), mp3[1] - width / 2 * np.cos(yaw)]
    p4 = [mp1[0] + width / 2 * np.sin(yaw), mp1[1] - width / 2 * np.cos(yaw)]

    px = [p1[0], p2[0], p3[0], p4[0]]
    py = [p1[1], p2[1], p3[1], p4[1]]

    px.append(px[0])
    py.append(py[0])

    return px, py
