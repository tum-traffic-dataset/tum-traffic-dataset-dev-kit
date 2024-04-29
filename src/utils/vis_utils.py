import ast
import json
import os
import sys
from math import radians
from typing import List

import cv2
import numpy as np
from bitarray import bitarray
from distinctipy import distinctipy
from matplotlib import cm, pyplot as plt
import matplotlib as mpl
import open3d as o3d

from scipy.spatial.transform import Rotation as R

from src.utils.detection import Detection
from src.utils.transformation import transform_base_to_lidar
from src.utils.perspective import Perspective
from src.utils.utils import get_cuboid_corners, id_to_class_name_mapping, class_name_to_id_mapping

import pickle

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200


class VisualizationUtils:
    def __init__(self):
        # TODO: automatically calculate num colors
        # s110
        # self.num_colors = 20
        # s40/s50
        self.num_colors = 100
        # cm = plt.get_cmap('gist_rainbow')
        # self.random_colors = [cm(1. * i / self.num_colors) for i in range(self.num_colors)]
        self.random_colors = distinctipy.get_colors(self.num_colors, pastel_factor=0.2)
        # load uuid to color mapping to have matching colors of one object in multiple frames
        self.track_id_to_color_mapping = {}
        if os.path.isfile("id_to_color_mapping.txt") and os.path.exists("id_to_color_mapping.txt"):
            with open("id_to_color_mapping.txt", "rb") as color_file:
                self.track_id_to_color_mapping = pickle.load(color_file)
        # else:
        #     open("id_to_color_mapping.txt", 'w').close()

    def __del__(self):
        if os.path.isfile("id_to_color_mapping.txt") and os.path.exists("id_to_color_mapping.txt"):
            with open("id_to_color_mapping.txt", "wb") as color_file:
                pickle.dump(self.track_id_to_color_mapping, color_file)

    def draw_line(self, img, start_point, end_point, color):
        cv2.line(img, start_point, end_point, color, 3)

    def get_projection_matrix(self, camera_id, lidar_id, boxes_coordinate_system_origin):
        # TODO: load calibration values from json calib files!!!
        projection_matrix = None
        if camera_id == "s040_camera_basler_north_16mm":
            projection_matrix = np.array(
                [[-4.19951586e+02, 2.89201626e+03, -2.48593165e+02, 2.36281908e+05],
                 [3.38518055e+01, -5.46540594e+01, -2.84424001e+03, 7.74758133e+03],
                 [-9.62740980e-01, 1.55067950e-01, -2.21548500e-01, 4.43080190e+02]])
        elif camera_id == "s040_camera_basler_north_50mm":
            projection_matrix = np.array([
                [
                    -535.802521,
                    9090.03991,
                    -37.2093177,
                    382610.364
                ],
                [
                    -47.7936014,
                    35.1841562,
                    -9031.53469,
                    97282.6614
                ],
                [
                    -0.995558291,
                    0.0755554009,
                    -0.0561700232,
                    465.221911
                ]
            ])
        elif camera_id == "s050_camera_basler_south_16mm":
            projection_matrix = np.array(
                [
                    [
                        953.327898,
                        -2815.07218,
                        -210.738663,
                        -15398.3725
                    ],
                    [
                        -31.2745828,
                        24.4015771,
                        -2864.62252,
                        23180.4175
                    ],
                    [
                        0.98093986,
                        0.00547866,
                        -0.19423435,
                        0.780686139
                    ]
                ])
        elif camera_id == "s050_camera_basler_south_50mm":
            projection_matrix = np.array(
                [
                    [
                        1250.42963,
                        -8808.77311,
                        -111.485694,
                        -45896.5209
                    ],
                    [
                        -55.8584708,
                        72.8689486,
                        -8823.5344,
                        73990.3221
                    ],
                    [
                        0.99541545,
                        0.0757193,
                        -0.05843522,
                        0.077429189
                    ]
                ]
            )
        elif camera_id == "s110_camera_basler_south1_8mm" and lidar_id == "s110_lidar_ouster_south":
            # R02 for TUMTraf-I (intersection dataset)
            # R04 for TUMTraf-V2X (V2X cooperative perception dataset)
            # TODO: make this configurable and load from parameters!!!
            release = "R02"
            if release == "R02":
                # projection matrix from s110_lidar_ouster_south to s110_camera_basler_south1
                # lidar->camera calibration
                # projection matrix for TUMTraf-I dataset (R02 release)
                projection_matrix = np.array(
                    [
                        [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
                        [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
                        [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
                    ],
                    dtype=float,
                )
            elif release == "R04":
                intrinsic_camera_matrix = np.array([[-1301.42, 0, 940.389], [0, -1299.94, 674.417], [0, 0, 1]])
                extrinsic_matrix = np.array(
                    [
                        [-0.41205, 0.910783, -0.0262516, 15.0787],
                        [0.453777, 0.230108, 0.860893, 2.52926],
                        [0.790127, 0.342818, -0.508109, 3.67868],
                    ],
                )
                projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix)
        elif camera_id == "s110_camera_basler_south2_8mm" and lidar_id == "s110_lidar_ouster_south":
            # optimized intrinsics (calibration lidar to base)
            intrinsic_camera_matrix = np.array(
                [[1315.56, 0, 969.353, 0.0], [0, 1368.35, 579.071, 0.0], [0, 0, 1, 0.0]], dtype=float
            )
            # manual calibration, optimizing intrinsics and extrinsics
            transformation_matrix_lidar_to_base = np.array(
                [
                    [0.247006, -0.955779, -0.15961, -16.8017],
                    [0.912112, 0.173713, 0.371316, 4.66979],
                    [-0.327169, -0.237299, 0.914685, 6.4602],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            # extrinsic base to south2 camera
            transformation_matrix_base_to_camera = np.array(
                [
                    [0.8924758822566284, 0.45096261644035174, -0.01093243630327495, 14.921784677055939],
                    [0.29913535165414396, -0.6097951995429897, -0.7339399539506467, 13.668310799382738],
                    [-0.3376460291207414, 0.6517534297474759, -0.679126369559744, -5.630430017833277],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            extrinsic_matrix_lidar_to_camera = np.matmul(
                transformation_matrix_base_to_camera, transformation_matrix_lidar_to_base
            )
            projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix_lidar_to_camera)
        elif camera_id == "s110_camera_basler_south1_8mm" and lidar_id == "s110_lidar_ouster_north":
            # NOTE: the projection matrix below only works for Release 02 of TUMTraf-I dataset
            # TODO: calculate projection matrix for Release 04 (TUMTraf-V2X dataset)
            release = "R02"
            if release == "R02":
                # optimized intrinsic (lidar to base calibration)
                intrinsic_camera_matrix = np.array(
                    [[1305.59, 0, 933.819, 0], [0, 1320.61, 609.602, 0], [0, 0, 1, 0]], dtype=float
                )

                # extrinsic lidar north to base
                transformation_matrix_lidar_to_base = np.array(
                    [
                        [-0.064419, -0.997922, 0.00169282, -2.08748],
                        [0.997875, -0.0644324, -0.00969147, 0.226579],
                        [0.0097804, 0.0010649, 0.999952, 8.29723],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )
                # extrinsic base to south1 camera
                transformation_matrix_base_to_camera = np.array(
                    [
                        [0.9530205584452789, -0.3026130702071279, 0.013309580025851253, 1.7732651490941862],
                        [-0.1291778833442192, -0.4457786636335154, -0.8857733668968741, 7.609039571774588],
                        [0.27397972486181504, 0.842440925400074, -0.4639271468406554, 4.047780978836272],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )
                extrinsic_matrix_lidar_to_camera = np.matmul(
                    transformation_matrix_base_to_camera, transformation_matrix_lidar_to_base
                )
                projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix_lidar_to_camera)
                print("projection_matrix", repr(projection_matrix))
                # correct projection matrix for R2
                projection_matrix = np.array([[290.06440167, -1522.65885503, -417.08293461, -398.03125035],
                                              [-88.96283956, 6.86258619, -1451.78013497, 454.22071755],
                                              [0.81846385, -0.32818492, -0.47160557, -0.18257704]])

                # NOTE: This is the wrong projection matrix from openlabel file.
                # projection_matrix = np.array([[2.98092042e+02, -1.48369084e+03, -4.45556647e+02, -1.15875824e+03],
                #                               [-8.52928965e+01, 6.41064053e-01, -1.40961433e+03, 3.38177149e+02],
                #                               [8.01700000e-01, -3.36123000e-01, -4.94264000e-01, -8.37352000e-01]])
                # TODO: Fix openlabel files with correct projection matrix in R2
            elif release == "R04":
                # TODO: calculate projection matrix for Release 04 (TUMTraf-V2X dataset)
                raise NotImplementedError("TODO: calculate projection matrix for Release 04 (TUMTraf-V2X dataset).")
                sys.exit(1)
        elif camera_id == "s110_camera_basler_south2_8mm" and lidar_id == "s110_lidar_ouster_north":
            # projection matrix from s110_lidar_ouster_north to s110_camera_basler_south2
            intrinsic_camera_matrix = np.array(
                [[1282.35, 0.0, 957.578, 0.0], [0.0, 1334.48, 563.157, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float
            )
            extrinsic_matrix = np.array(
                [
                    [0.37383, -0.927155, 0.0251845, 14.2181],
                    [-0.302544, -0.147564, -0.941643, 3.50648],
                    [0.876766, 0.344395, -0.335669, -7.26891],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix)
            # [[ 1318.95273325,  -859.15213894,  -289.13390611, 11272.03223502],
            #        [   90.01799314,    -2.9727517 , -1445.63809767,   585.78988153],
            #        [    0.876766  ,     0.344395  ,    -0.335669  ,    -7.26891   ]])

            # from label file
            # projection_matrix = np.array([[1.31895273e+03, -8.59152139e+02, -2.89133906e+02,1.12720322e+04],
            #                               [9.00179931e+01, -2.97275170e+00, -1.44563810e+03,5.85789882e+02],
            #                               [8.76766000e-01, 3.44395000e-01, -3.35669000e-01,-7.26891000e+00]])
        elif camera_id == "s110_camera_basler_east_8mm" and lidar_id == "s110_lidar_ouster_south":
            projection_matrix = np.array(
                [
                    [-2666.70160799, -655.44528859, -790.96345758, -33010.77350141],
                    [430.89231274, 66.06703744, -2053.70223986, 6630.65222157],
                    [-0.00932524, -0.96164431, -0.27414094, 11.41820108],
                ]
            )
        elif camera_id == "s110_camera_basler_north_8mm" and lidar_id == "s110_lidar_ouster_south":
            intrinsic_matrix = np.array([[1360.68, 0, 849.369], [0, 1470.71, 632.174], [0, 0, 1]])
            extrinsic_matrix = np.array(
                [
                    [-0.564602, -0.824833, -0.0295815, -12.9358],
                    [-0.458346, 0.343143, -0.819861, 7.22666],
                    [0.686399, -0.449337, -0.571798, -6.75018],
                ],
            )
            projection_matrix = np.matmul(intrinsic_matrix, extrinsic_matrix)
        elif camera_id == "vehicle_camera_basler_16mm" and lidar_id == "vehicle_lidar_robosense":
            projection_matrix = np.array(
                [[1019.929965441548, -2613.286262078907, 184.6794570200418, 370.7180273597151],
                 [589.8963703919744, -24.09642935106967, -2623.908527352794, -139.3143336725661],
                 [0.9841844439506531, 0.1303769648075104, 0.1199281811714172, -0.1664766669273376]])

        else:
            print("Error. Unknown camera/LiDAR ID passed: camera_id=", camera_id, ", lidar_id=", lidar_id,
                  ". Exiting...")
            sys.exit()

        if boxes_coordinate_system_origin == "s110_base" and lidar_id != "vehicle_lidar_robosense":
            if camera_id == "s110_camera_basler_south1_8mm":
                # projection matrix from s110_base to s110_camera_basler_south1_8mm (camera to hd map calibration)
                projection_matrix = np.array(
                    [
                        [1599.6787257016188, 391.55387236603775, -430.34650625835917, 6400.522155319611],
                        [-21.862527625533737, -135.38146150648188, -1512.651893582593, 13030.4682633739],
                        [0.27397972486181504, 0.842440925400074, -0.4639271468406554, 4.047780978836272],
                    ],
                    dtype=float,
                )
            elif camera_id == "s110_camera_basler_south2_8mm":
                projection_matrix = np.array(
                    [
                        [587.0282580208467, 1104.2087993634066, -678.1758072601377, 9829.430183087894],
                        [-45.53930633847783, 51.56632485337586, -1590.519015462949, 8982.057054635612],
                        [-0.3376460291207414, 0.6517534297474759, -0.679126369559744, -5.630430017833277],
                    ]
                )
            elif camera_id == "vehicle_camera_basler_16mm":
                intrinsic_camera_matrix = np.array([
                    [
                        2788.86072,
                        0,
                        907.839058
                    ],
                    [
                        0,
                        2783.31261,
                        589.071478
                    ],
                    [
                        0,
                        0,
                        1
                    ]
                ])
                extrinsic_matrix_s110_base_to_vehicle_camera = np.array([
                    [
                        -0.10853248450718671,
                        0.9802219683449281,
                        0.16548753616347203,
                        17.58632616863186
                    ],
                    [
                        -0.05018854624653521,
                        -0.17166166576987898,
                        0.9838771024765322,
                        27.38400788218499
                    ],
                    [
                        0.9928247589397987,
                        0.0984767159520325,
                        0.06782674254885492,
                        -0.023219830174785155
                    ]
                ])
                projection_matrix = np.matmul(intrinsic_camera_matrix, extrinsic_matrix_s110_base_to_vehicle_camera)
            else:
                print("Error. Unknown camera passed: ", camera_id, ". Exiting...")
                sys.exit()
        return projection_matrix

    def hex_to_rgb(self, value):
        value = value.lstrip("#")
        lv = len(value)
        return tuple(int(value[i: i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def draw_2d_box(self, img, box_label, color):
        cv2.rectangle(img, (box_label[0], box_label[1]), (box_label[2], box_label[3]), color, 1)

    def project_point_cloud_to_image(self, image, points_3d, lidar_id, camera_id):
        overlay = image.copy()

        # remove rows having all zeros (131k points -> 59973 points)
        points_3d = points_3d[~np.all(points_3d == 0, axis=1)]

        # crop point cloud to 120 m range
        distances = np.array([np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2) for point in points_3d])
        points_3d = points_3d[distances < 120.0]

        points_3d = np.transpose(points_3d)
        points_3d = np.append(points_3d, np.ones((1, points_3d.shape[1])), axis=0)
        distances = []
        indices_to_keep = []
        for i in range(len(points_3d[0, :])):
            point = points_3d[:, i]
            distance = np.sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2))
            if distance > 2:
                distances.append(distance)
                indices_to_keep.append(i)

        points_3d = points_3d[:, indices_to_keep]

        # project points to 2D
        projection_matrix = self.get_projection_matrix(camera_id, lidar_id,
                                                       boxes_coordinate_system_origin="s110_lidar_ouster_south")
        points = np.matmul(projection_matrix, points_3d[:4, :])
        distances_numpy = np.asarray(distances)
        max_distance = max(distances_numpy)
        norm = mpl.colors.Normalize(vmin=70, vmax=250)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        num_points_within_image = 0
        for i in range(len(points[0, :])):
            if points[2, i] > 0:
                pos_x = int(points[0, i] / points[2, i])
                pos_y = int(points[1, i] / points[2, i])
                if pos_x >= 0 and pos_x < 1920 and pos_y >= 0 and pos_y < 1200:
                    num_points_within_image += 1
                    distance_idx = 255 - (int(distances_numpy[i] / max_distance * 255))
                    color_rgba = m.to_rgba(distance_idx)
                    color_rgb = (
                        color_rgba[0] * 255,
                        color_rgba[1] * 255,
                        color_rgba[2] * 255,
                    )
                    cv2.circle(overlay, (pos_x, pos_y), 4, color_rgb, thickness=-1)

        print("num points within image: ", num_points_within_image)
        alpha = 0.7
        image_with_transparent_points = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return image_with_transparent_points

    def project_3d_box_to_2d(self, points_3d, camera_id, lidar_id, boxes_coordinate_system_origin):
        points_3d = np.transpose(points_3d)

        if boxes_coordinate_system_origin == "s040_camera_basler_north_16mm":
            # just multiply with intrinsic because the objects are already in camera space
            intrinsic_matrix = np.array([[2.78886072e+03, 0.00000000e+00, 9.07839058e+02],
                                         [0.00000000e+00, 2.78331261e+03, 5.89071478e+02],
                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)
            # multiply with intrinsic matrix
            points = np.matmul(intrinsic_matrix, points_3d[:3, :])
        elif boxes_coordinate_system_origin == "s040_camera_basler_north_50mm":
            intrinsic_matrix = np.array([[9.02348282e+03, 0.00000000e+00, 1.22231430e+03],
                                         [0.00000000e+00, 9.01450436e+03, 5.57541182e+02],
                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)
            # multiply with intrinsic matrix
            points = np.matmul(intrinsic_matrix, points_3d[:3, :])
        elif boxes_coordinate_system_origin == "s050_camera_basler_south_16mm":
            intrinsic_matrix = np.array([[2.82046004e+03, 0.00000000e+00, 9.60667182e+02],
                                         [0.00000000e+00, 2.81622151e+03, 5.25863289e+02],
                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)
            # multiply with intrinsic matrix
            points = np.matmul(intrinsic_matrix, points_3d[:3, :])
        elif boxes_coordinate_system_origin == "s050_camera_basler_south_50mm":
            intrinsic_matrix = np.array([[8.87857970e+03, 0.00000000e+00, 5.84217565e+02],
                                         [0.00000000e+00, 8.81172402e+03, 4.65520403e+02],
                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)
            # multiply with intrinsic matrix
            points = np.matmul(intrinsic_matrix, points_3d[:3, :])
        else:
            points_3d = np.append(points_3d, np.ones((1, points_3d.shape[1])), axis=0)
            # project points to 2D
            projection_matrix = self.get_projection_matrix(camera_id, lidar_id, boxes_coordinate_system_origin)
            points = np.matmul(projection_matrix, points_3d[:4, :])
        # filter out points behind camera
        points = points[:, points[2] > 0]
        # return None if no points are visible
        if points.shape[1] == 0:
            return None
        # Divide x and y values by z (camera pinhole model).
        image_points = points[:2] / points[2]

        return image_points

    def draw_3d_boxes(
            self,
            image: np.ndarray,
            detections: List[Detection],
            perspective: Perspective,
            box_thickness: float = 2,
            draw_3d_center_position=False,
            draw_orientation=False,
            draw_tracking_info=True,
            input_type: str = None,
            use_two_colors: bool = False,
    ):
        """Draw detections as 3D box on image."""
        for detection_index, detection in enumerate(detections):

            if use_two_colors and input_type == "detections":
                color_rgb = (245, 44, 71)  # red
            elif use_two_colors and input_type == "labels":
                color_rgb = (27, 250, 27)  # green
            else:
                color_rgb = id_to_class_name_mapping[str(class_name_to_id_mapping[detection.category])]["color_rgb"]

            # swap channels because opencv uses bgr
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Create local bounding box.
            length, width, height = detection.dimensions
            bottom_corners = np.array(
                [
                    [-length / 2, +length / 2, +length / 2, -length / 2],
                    [-width / 2, -width / 2, +width / 2, +width / 2],
                    [0, 0, 0, 0],
                ]
            )
            # Rotate bounding box.
            yaw = detection.yaw
            rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            bottom_corners = rotation @ bottom_corners

            # Add global location, top corners and project onto image.
            height_offset = np.array([[0, 0, height]]).T
            bottom_corners += detection.location
            corners = np.hstack((bottom_corners, bottom_corners + height_offset))
            projected_corners = perspective.project_from_base_to_image(corners).astype(int).T
            projected_front_center = ((projected_corners[0] + projected_corners[3]) * 0.5).flatten()[:2].astype(int)
            projected_center = perspective.project_from_base_to_image(detection.location).flatten()[:2].astype(int)

            if draw_orientation:
                cv2.arrowedLine(image, projected_center, projected_front_center, (0, 255, 255), 1, cv2.LINE_AA)

            # Add bottom-, top polygons and vertical lines individually.
            polygons = [projected_corners[:4], projected_corners[4:]]
            for j in range(4):
                polygons.append(projected_corners[[j, j + 4]])

            # Draw polygons.
            cv2.polylines(img=image, pts=polygons, isClosed=True, color=color_bgr, thickness=box_thickness)

            # Draw additional annotations
            if draw_3d_center_position:
                cv2.circle(image, projected_center, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
            if draw_tracking_info:
                text = f"{detection.category} "
                if detection.id > 0:
                    text += str(detection.id)
                else:
                    text += str(detection_index)
                if detection.velocity is not None:
                    text += f" ({detection.velocity[0]:.2f}, {detection.velocity[1]:.2f})"
                cv2.rectangle(
                    image,
                    projected_center + [-2, 3],
                    projected_center + [5 * len(text) + 2, -10],
                    (0, 0, 0),
                    cv2.FILLED,
                )
                cv2.putText(
                    image, text, projected_center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA
                )

    def draw_3d_box(self, img, box_label, color, camera_id, lidar_id, boxes_coordinate_system_origin, perspective,
                    dataset_release):
        if boxes_coordinate_system_origin == "s110_base" and lidar_id != "vehicle_lidar_robosense":
            l = float(box_label["object_data"]["cuboid"]["val"][7])
            w = float(box_label["object_data"]["cuboid"]["val"][8])
            h = float(box_label["object_data"]["cuboid"]["val"][9])
            quat_x = float(box_label["object_data"]["cuboid"]["val"][3])
            quat_y = float(box_label["object_data"]["cuboid"]["val"][4])
            quat_z = float(box_label["object_data"]["cuboid"]["val"][5])
            quat_w = float(box_label["object_data"]["cuboid"]["val"][6])
            roll, pitch, yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("xyz", degrees=True)
            location = np.array(
                [
                    [float(box_label["object_data"]["cuboid"]["val"][0])],
                    [float(box_label["object_data"]["cuboid"]["val"][1])],
                    [float(box_label["object_data"]["cuboid"]["val"][2])],
                ]
            )
            if dataset_release == "r00":
                # add half of the height to the z coordinate
                location[2] += h / 2
            yaw = radians(yaw)
            bottom_corners = np.array(
                [
                    [-l / 2, +l / 2, +l / 2, -l / 2],
                    [-w / 2, -w / 2, +w / 2, +w / 2],
                    [0, 0, 0, 0],
                ]
            )
            # Rotate bounding box.
            rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            bottom_corners = rotation @ bottom_corners
            # Add global location, top corners and project onto image.
            height_offset = np.array([[0, 0, h]]).T
            bottom_corners += location
            corners = np.hstack((bottom_corners, bottom_corners + height_offset))
            points_2d = perspective.project_from_base_to_image(corners, filter_behind=True).astype(int)
        else:
            points_3d = get_cuboid_corners(box_label["object_data"]["cuboid"]["val"],
                                           boxes_coordinate_system_origin=boxes_coordinate_system_origin)
            points_2d = self.project_3d_box_to_2d(points_3d, camera_id, lidar_id, boxes_coordinate_system_origin)

        if points_2d is None:
            return None
        if points_2d.shape[0] == 2 and points_2d.shape[1] > 1:
            points_2d = np.array(points_2d, dtype=int)
            points_2d = points_2d.T

            if "attributes" in box_label["object_data"]["cuboid"]:
                attribute = self.get_attribute_by_name(
                    box_label["object_data"]["cuboid"]["attributes"]["num"], "num_points"
                )
                if attribute is not None:
                    num_points = int(attribute["val"])
                    # label += "," + str(num_points)
                attribute = self.get_attribute_by_name(box_label["object_data"]["cuboid"]["attributes"]["num"], "score")
                if attribute is not None:
                    score = round(float(attribute["val"]), 2)
                    # label += "," + str(score)
                attribute = self.get_attribute_by_name(
                    box_label["object_data"]["cuboid"]["attributes"]["text"], "overlap"
                )
                if attribute is not None:
                    overlap = attribute["val"]
                    # label += "," + str(overlap)

            # mark front size with cross
            if points_2d.shape[0] == 8 and points_2d.shape[1] == 2:
                polygons = [points_2d[:4], points_2d[4:]]
                for j in range(4):
                    polygons.append(points_2d[[j, j + 4]])

                # Draw polygons.
                cv2.polylines(img=img, pts=polygons, isClosed=True, color=color, thickness=2)

                # visualize front face with cross (X)
                if boxes_coordinate_system_origin == "s110_base":
                    self.draw_line(img, points_2d[0], points_2d[7], color)
                    self.draw_line(img, points_2d[3], points_2d[4], color)
                else:
                    self.draw_line(img, points_2d[2], points_2d[7], color)
                    self.draw_line(img, points_2d[3], points_2d[6], color)
        return points_2d

    @staticmethod
    def get_attribute_by_name(attribute_list, attribute_name):
        for attribute in attribute_list:
            if attribute["name"] == attribute_name:
                return attribute
        return None

    def visualize_bounding_box(
            self, l, w, h, rotation_yaw, position_3d, category, use_two_colors, input_type, renderer, material,
            object_id
    ):
        quaternion = R.from_euler("xyz", [0, 0, rotation_yaw], degrees=False).as_quat()
        corner_box = get_cuboid_corners(
            [
                position_3d[0],
                position_3d[1],
                position_3d[2],
                quaternion[0],
                quaternion[1],
                quaternion[2],
                quaternion[3],
                l,
                w,
                h,
            ]
        )
        # colors = [[1, 0, 0] for _ in range(len(lines))]
        line_indices = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
        if use_two_colors and input_type == "detections":
            color_red_rgb = (245, 44, 71)
            # color_red_rgb_normalized = (color_red_rgb[0] / 255, color_red_rgb[1] / 255, color_red_rgb[2] / 255)
            color_red_bgr_normalized = (color_red_rgb[2] / 255, color_red_rgb[1] / 255, color_red_rgb[0] / 255)
            colors = [color_red_bgr_normalized for _ in range(len(line_indices))]
        elif use_two_colors and input_type == "labels":
            color_green_rgb = (27, 250, 27)
            # color_green_normalized = (color_green_rgb[0] / 255, color_green_rgb[1] / 255, color_green_rgb[2] / 255)
            color_green_bgr_normalized = (color_green_rgb[2] / 255, color_green_rgb[1] / 255, color_green_rgb[0] / 255)
            colors = [color_green_bgr_normalized for _ in range(len(line_indices))]
        else:
            # change from rgb to bgr
            colors = [
                id_to_class_name_mapping[str(class_name_to_id_mapping[category])]["color_bgr_normalized"]
                for _ in range(len(line_indices))
            ]

        line_set = o3d.geometry.LineSet()

        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # Display the bounding boxes:
        renderer.scene.add_geometry("line_set_" + object_id, line_set, material["line"])
        # TODO: color front site/face of 3D box in color (2 colored triangles)
        # renderer.scene.remove_geometry("line_set_" + object_id)
        return line_set

    def visualize_track_history(self, track_history, category, use_two_colors, input_type, renderer, material,
                                object_id):
        # round all position values inside track_history to 3 decimal places
        track_history = np.around(track_history, decimals=3)
        # remove duplicates from track history but keep order
        track_history = np.array(list(dict.fromkeys(map(tuple, track_history))))
        if track_history.shape[0] < 2:
            return
        if use_two_colors and input_type == "detections":
            color_red_rgb = (245, 44, 71)
            # color_red_rgb_normalized = (color_red_rgb[0] / 255, color_red_rgb[1] / 255, color_red_rgb[2] / 255)
            color_red_bgr_normalized = (color_red_rgb[2] / 255, color_red_rgb[1] / 255, color_red_rgb[0] / 255)
            colors = [color_red_bgr_normalized for _ in range(len(track_history) - 1)]
        elif use_two_colors and input_type == "labels":
            color_green_rgb = (27, 250, 27)
            # color_green_normalized = (color_green_rgb[0] / 255, color_green_rgb[1] / 255, color_green_rgb[2] / 255)
            color_green_bgr_normalized = (color_green_rgb[2] / 255, color_green_rgb[1] / 255, color_green_rgb[0] / 255)
            colors = [color_green_bgr_normalized for _ in range(len(track_history) - 1)]
        else:
            # change from rgb to bgr
            colors = [
                id_to_class_name_mapping[str(class_name_to_id_mapping[category])]["color_bgr_normalized"]
                for _ in range(len(track_history) - 1)
            ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(track_history)
        line_indices = [[i, i + 1] for i in range(len(track_history) - 1)]
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        renderer.scene.add_geometry("track_history_line_set_" + object_id, line_set, material["line"])
        # TODO: color front site/face of 3D box in color (2 colored triangles)
        # renderer.scene.remove_geometry("line_set_" + object_id)
        return line_set

    def visualize_boxes_3d(
            self,
            pcd,
            file_path_point_cloud,
            file_path_labels,
            file_path_detections,
            use_detections_in_base,
            renderer,
            material,
    ):
        points = np.array(pcd.points)
        # remove rows having all zeroes
        points_filtered = points[~np.all(points == 0, axis=1)]

        # remove rows having z>-1.5 (for ouster lidars)
        if "ouster" in file_path_point_cloud:
            points_filtered = points_filtered[points_filtered[:, 2] < 2]

        # remove rows having z<-10
        points_filtered = points_filtered[points_filtered[:, 2] > -10.0]

        # remove points with distance>120
        distances = np.array([np.sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]) for row in points_filtered])
        points_filtered = points_filtered[distances < 120.0]
        distances = distances[distances < 120.0]

        # remove points with distance<3
        points_filtered = points_filtered[distances > 3.0]
        corner_point_min = np.array([-150, -150, -10])
        corner_point_max = np.array([150, 150, 5])
        points = np.vstack((points_filtered, corner_point_min, corner_point_max))

        # if use_detections_in_base:
        #     points_transformed = transform_into_base(points)

        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[:, :3]))
        color_background_point_cloud = False
        if color_background_point_cloud:
            pcd = self.color_point_cloud(pcd)
        else:
            pcd.paint_uniform_color([0.1, 0.1, 0.1])

        use_two_colors = file_path_labels != "" and file_path_detections != ""
        object_id_list = []
        label_object_id_list = []
        detection_object_id_list = []
        if file_path_labels != "":
            label_data = json.load(open(file_path_labels))
            label_object_id_list = self.process_data(
                label_data,
                pcd,
                use_two_colors,
                input_type="labels",
                use_detections_in_base=use_detections_in_base,
                renderer=renderer,
                material=material,
            )
        if file_path_detections != "":
            detection_data = json.load(open(file_path_detections))
            detection_object_id_list = self.process_data(
                detection_data,
                pcd,
                use_two_colors,
                input_type="detections",
                use_detections_in_base=use_detections_in_base,
                renderer=renderer,
                material=material,
            )
        object_id_list = label_object_id_list + detection_object_id_list

        renderer.scene.add_geometry("pcd", pcd, material["point"])
        return object_id_list

    def plot_banner(self, image, x_min, y_min, x_max, y_max, color, label):
        line_height = y_max - y_min
        x_middle = int(x_min + ((x_max - x_min) / 2.0))
        y_middle = int(y_min + ((y_max - y_min) / 2.0))
        image = cv2.line(image, (x_middle, y_middle), (x_middle, y_middle - int(line_height / 2.0)), color, 2)
        image = cv2.circle(image, (x_middle, y_middle), radius=4, color=color, thickness=-1)
        t_size = cv2.getTextSize(label, 0, fontScale=1.0, thickness=1)[0]
        c2 = x_min + int((x_max - x_min) / 2.0) + t_size[0], y_middle - int(line_height / 2.0) - t_size[1] - 3
        image = cv2.rectangle(
            image, (x_min + int((x_max - x_min) / 2.0), y_middle - int(line_height / 2.0)), c2, color, -1, cv2.LINE_AA
        )  # filled
        image = cv2.putText(
            image,
            label,
            (x_min + int((x_max - x_min) / 2.0), y_middle - int(line_height / 2.0) - 2),
            0,
            1.0,
            [255, 255, 255],
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return image

    def process_data(self, box_data, pcd, use_two_colors, input_type, use_detections_in_base, renderer, material):
        object_id_list = []
        # line_set_list = []
        if "openlabel" in box_data:
            for frame_id, frame_obj in box_data["openlabel"]["frames"].items():
                for object_id, label in frame_obj["objects"].items():
                    object_id_list.append(object_id)
                    # Dataset in ASAM OpenLABEL format
                    l = float(label["object_data"]["cuboid"]["val"][7])
                    w = float(label["object_data"]["cuboid"]["val"][8])
                    h = float(label["object_data"]["cuboid"]["val"][9])
                    quat_x = float(label["object_data"]["cuboid"]["val"][3])
                    quat_y = float(label["object_data"]["cuboid"]["val"][4])
                    quat_z = float(label["object_data"]["cuboid"]["val"][5])
                    quat_w = float(label["object_data"]["cuboid"]["val"][6])
                    rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("xyz", degrees=False)[2]
                    position_3d = [
                        float(label["object_data"]["cuboid"]["val"][0]),
                        float(label["object_data"]["cuboid"]["val"][1]),
                        float(label["object_data"]["cuboid"]["val"][2]),
                    ]
                    category = label["object_data"]["type"].upper()
                    # paint points within bounding box by class color

                    # transform labels to s110_base
                    # if use_detections_in_base and input_type == "labels":
                    #     position_3d = transform_into_base(np.array(position_3d).reshape(1, 3)).T.flatten()

                    # transform detections from s110_base to lidar
                    if use_detections_in_base and input_type == "detections":
                        rotation_yaw = rotation_yaw + np.deg2rad(103)
                        position_3d = transform_base_to_lidar(np.array(position_3d).reshape(1, 3)).T.flatten()

                    obb = o3d.geometry.OrientedBoundingBox(
                        position_3d,
                        np.array(
                            [
                                [np.cos(rotation_yaw), -np.sin(rotation_yaw), 0],
                                [np.sin(rotation_yaw), np.cos(rotation_yaw), 0],
                                [0, 0, 1],
                            ]
                        ),
                        np.array([l, w, h]),
                    )
                    indices = obb.get_point_indices_within_bounding_box(pcd.points)
                    colors = np.array(pcd.colors)

                    if use_two_colors and input_type == "detections":
                        color_red_rgb = (245, 44, 71)
                        colors[indices, :] = np.array(
                            [color_red_rgb[2] / 255, color_red_rgb[1] / 255, color_red_rgb[0] / 255]
                        )
                    elif use_two_colors and input_type == "labels":
                        color_green_rgb = (27, 250, 27)
                        colors[indices, :] = np.array(
                            [color_green_rgb[2] / 255, color_green_rgb[1] / 255, color_green_rgb[0] / 255]
                        )
                    else:
                        class_color_rgb = id_to_class_name_mapping[str(class_name_to_id_mapping[category.upper()])][
                            "color_rgb_normalized"
                        ]
                        # class_color_bgr = (class_color_rgb[2], class_color_rgb[1], class_color_rgb[0])
                        colors[indices] = np.array([class_color_rgb[2], class_color_rgb[1], class_color_rgb[0]])
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    # box_cloud = pcd.select_by_index(indices)
                    # box_cloud..paint_uniform_color([0.4, 0.4, 0.4])
                    # position_3d[2] = position_3d[2] - h / 2  # To avoid floating bounding boxes
                    self.visualize_bounding_box(
                        l,
                        w,
                        h,
                        rotation_yaw,
                        position_3d,
                        category,
                        use_two_colors,
                        input_type,
                        renderer,
                        material,
                        object_id,
                    )

                    if "attributes" in label["object_data"]["cuboid"]:
                        if "vec" in label["object_data"]["cuboid"]["attributes"]:
                            attribute = VisualizationUtils.get_attribute_by_name(
                                label["object_data"]["cuboid"]["attributes"]["vec"], "track_history"
                            )
                            if attribute is not None:
                                track_history_list = attribute["val"]
                                pos_history = np.array(track_history_list).reshape(-1, 3)
                            else:
                                pos_history = np.array([])

                            if len(pos_history) > 1:
                                self.visualize_track_history(pos_history, category, use_two_colors, input_type,
                                                             renderer,
                                                             material, object_id)
        else:
            for label in box_data["labels"]:
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
                category = label["category"].upper()
                self.visualize_bounding_box(l, w, h, rotation, position_3d, category)
        return object_id_list

    def hex_to_rgb(self, value):
        value = value.lstrip("#")
        lv = len(value)
        return tuple(int(value[i: i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def color_point_cloud(self, pcd):
        sorted_z = np.asarray(pcd.points)[np.argsort(np.asarray(pcd.points)[:, 2])[::-1]]
        rows = len(pcd.points)
        pcd.normalize_normals()
        # when Z values are negative, this if else statement switches the min and max
        if sorted_z[0][2] < sorted_z[rows - 1][2]:
            min_z_val = sorted_z[0][2]
            max_z_val = sorted_z[rows - 1][2]
        else:
            max_z_val = sorted_z[0][2]
            min_z_val = sorted_z[rows - 1][2]
        # assign colors to the point cloud file
        cmap_norm = mpl.colors.Normalize(vmin=min_z_val, vmax=max_z_val)
        # example color maps: jet, hsv.  Further colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        point_colors = plt.get_cmap("jet")(cmap_norm(np.asarray(pcd.points)[:, -1]))[:, 0:3]
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        return pcd

    def draw_2d_box(self, img, box_label, color, line_width):
        cv2.rectangle(img, (box_label[0], box_label[1]), (box_label[2], box_label[3]), color, line_width)

    def draw_3d_box_openlabel(self, img, box_label, color):
        pass
        # NOTE: coordinates are NOT normalized
        # TODO: change to OpenLABEL order
        # (0): top_left_front
        # (1): top_right_front
        # (2): bottom_right_front
        # (3): bottom_left_front
        # (4): top_left_back
        # (5): top_right_back
        # (6): bottom_right_back
        # (7): bottom_left_back

        #   (4) o--------o(5)
        #     /|       /|
        #    / |      / |
        # (0)o--------o  |(1)
        #   |(7)o-----|--o(6)
        #   | /      | /
        #   |/       |/
        # (3)o--------o(2)

    def draw_pseudo_3d_box(self, img, box, color, use_normalized_keypoints=True):
        # NOTE: coordinates are normalized
        # (0): bottom_left_front
        # (1): bottom_left_back
        # (2): bottom_right_back
        # (3): bottom_right_front
        # (4): top_left_front
        # (5): top_left_back
        # (6): top_right_back
        # (7): top_right_front

        #   (5) o--------o(6)
        #     /|       /|
        #    / |      / |
        # (4)o--------o  |(7)
        #   |(1)o-----|--o(2)
        #   | /      | /
        #   |/       |/
        # (0)o--------o(3)

        if use_normalized_keypoints:
            w = IMAGE_WIDTH
            h = IMAGE_HEIGHT
        else:
            w = 1.0
            h = 1.0

        # draw bottom 4 lines
        # bottom_left_front -> bottom_left_back
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][0]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][0]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][1]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][1]["point2d"]["val"][1] * h),
            ),
            color,
        )
        # bottom_left_back -> bottom_right_back
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][1]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][1]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][2]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][2]["point2d"]["val"][1] * h),
            ),
            color,
        )
        # bottom_right_back -> bottom_right_front
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][2]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][2]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][3]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][3]["point2d"]["val"][1] * h),
            ),
            color,
        )
        # bottom_right_front -> bottom_left_front
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][3]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][3]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][0]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][0]["point2d"]["val"][1] * h),
            ),
            color,
        )

        # draw top 4 lines
        # top_left_front -> top_left_back
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][4]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][4]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][5]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][5]["point2d"]["val"][1] * h),
            ),
            color,
        )
        # top_left_back -> top_right_back
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][5]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][5]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][6]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][6]["point2d"]["val"][1] * h),
            ),
            color,
        )
        # top_right_back -> top_right_front
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][6]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][6]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][7]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][7]["point2d"]["val"][1] * h),
            ),
            color,
        )
        # top_right_front -> top_left_front
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][7]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][7]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][4]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][4]["point2d"]["val"][1] * h),
            ),
            color,
        )

        # draw 4 vertical lines
        # bottom_left_front -> top_left_front
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][0]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][0]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][4]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][4]["point2d"]["val"][1] * h),
            ),
            color,
        )
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][1]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][1]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][5]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][5]["point2d"]["val"][1] * h),
            ),
            color,
        )
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][2]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][2]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][6]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][6]["point2d"]["val"][1] * h),
            ),
            color,
        )
        self.draw_line(
            img,
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][3]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][3]["point2d"]["val"][1] * h),
            ),
            (
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][7]["point2d"]["val"][0] * w),
                int(box["keypoints_2d"]["attributes"]["points2d"]["val"][7]["point2d"]["val"][1] * h),
            ),
            color,
        )

        # draw front face marked with a cross (2 lines)
        # top_right_back -> bottom_right_front
        # self.draw_line(
        #     img,
        #     (
        #         int(box["keypoints_2d"]["attributes"]["points2d"]["val"][0]["point2d"]["val"][0] * w),
        #         int(box["keypoints_2d"]["attributes"]["points2d"]["val"][0]["point2d"]["val"][1] * h),
        #     ),
        #     (
        #         int(box["keypoints_2d"]["attributes"]["points2d"]["val"][7]["point2d"]["val"][0] * w),
        #         int(box["keypoints_2d"]["attributes"]["points2d"]["val"][7]["point2d"]["val"][1] * h),
        #     ),
        #     color,
        # )
        #
        # self.draw_line(
        #     img,
        #     (
        #         int(box["keypoints_2d"]["attributes"]["points2d"]["val"][4]["point2d"]["val"][0] * w),
        #         int(box["keypoints_2d"]["attributes"]["points2d"]["val"][4]["point2d"]["val"][1] * h),
        #     ),
        #     (
        #         int(box["keypoints_2d"]["attributes"]["points2d"]["val"][3]["point2d"]["val"][0] * w),
        #         int(box["keypoints_2d"]["attributes"]["points2d"]["val"][3]["point2d"]["val"][1] * h),
        #     ),
        #     color,
        # )

    def project_3d_position(self, position_3d):
        # project 3D Position into camera image
        # TODO: load projection matrix from A9 dataset (label json files)
        # from s110_base into s110_camera_basler_south1_8mm
        projection_matrix = np.array(
            [
                [1599.6787257016188, 391.55387236603775, -430.34650625835917, 6400.522155319611],
                [-21.862527625533737, -135.38146150648188, -1512.651893582593, 13030.4682633739],
                [0.27397972486181504, 0.842440925400074, -0.4639271468406554, 4.047780978836272],
            ],
            dtype=float,
        )

        # from s110_base into s110_camera_basler_south2_8mm (from camera-lidar calibration)
        # projection_matrix = np.array([
        #     [
        #         587.0282580208467,
        #         1104.2087993634066,
        #         -678.1758072601377,
        #         9829.430183087894
        #     ],
        #     [
        #         -45.53930633847783,
        #         51.56632485337586,
        #         -1590.519015462949,
        #         8982.057054635612
        #     ],
        #     [
        #         -0.3376460291207414,
        #         0.6517534297474759,
        #         -0.679126369559744,
        #         -5.630430017833277
        #     ]
        # ], dtype=float)

        position_3d_homogeneous = np.array([position_3d[0], position_3d[1], position_3d[2], 1])
        points = np.matmul(projection_matrix, position_3d_homogeneous)
        if points[2] > 0:
            pos_x = int(points[0] / points[2])
            pos_y = int(points[1] / points[2])
            # if pos_x >= 0 and pos_x < 1920 and pos_y >= 0 and pos_y < 1200:
            return [pos_x, pos_y]
        return None

    def draw_mask(self, img, mask, color_bgr, mask_height, mask_width):
        buffer = ast.literal_eval(mask)
        bit_array = bitarray(buffer=buffer, endian="big")
        mask = np.frombuffer(bit_array.unpack(), dtype=bool)
        mask = mask.reshape((mask_height, mask_width))

        padding = int((mask_height - img.shape[0]) / 2)
        mask = np.array(mask)[padding:-padding, :]

        img_mask_np = np.zeros(mask.shape, dtype=bool)
        img_mask_np[mask > 0] = 1
        img_mask_np = img_mask_np.astype(bool)

        img[img_mask_np] = img[img_mask_np] // 3 * 2 + (np.array(color_bgr, dtype="uint8") // 3)

    def check_within_image(self, box3d):
        corners = [
            "bottom_left_front",
            "bottom_right_front",
            "top_left_front",
            "top_right_front",
            "bottom_left_back",
            "bottom_right_back",
            "top_left_back",
            "top_right_back",
        ]
        for corner in corners:
            if box3d[corner] is None:
                continue
            if (
                    box3d[corner][0] >= 0
                    and box3d[corner][0] < IMAGE_WIDTH
                    and box3d[corner][1] >= 0
                    and box3d[corner][1] < IMAGE_HEIGHT
            ):
                return True
        return False

    def transform_from_s110_base_to_s110_lidar_ouster_south(self, location: np.ndarray):
        """Transform location from s110 base frame to s110 lidar ouster south frame."""
        transformation_matrix_s110_lidar_ouster_south_to_s110_base = np.array([
            [
                0.21479485,
                -0.9761028,
                0.03296187,
                -15.87257873
            ],
            [
                0.97627128,
                0.21553835,
                0.02091894,
                2.30019086
            ],
            [
                -0.02752358,
                0.02768645,
                0.99923767,
                7.48077521
            ],
            [
                0.00000000,
                0.00000000,
                0.00000000,
                1.00000000
            ]
        ], dtype=float)

        location_transformed = transformation_matrix_s110_lidar_ouster_south_to_s110_base @ np.append(location, 1)
        return location_transformed[:3]

    def plot_statistics(self, img, num_detections_lidar, num_detections_camera, num_detections_fused):

        # calculate height (in pixel) of gray rectangle
        height = 0
        if num_detections_lidar > 0:
            height += 50
        if num_detections_camera > 0:
            height += 50
        if num_detections_fused > 0:
            height += 50
            height += 50

        # plot number of monocular detections
        shapes = np.zeros_like(img, np.uint8)
        cv2.rectangle(shapes, (0, 0), (400, height), (96, 96, 96), cv2.FILLED)

        img_copy = img.copy()
        alpha = 0.4
        mask = shapes.astype(bool)
        # cv2.rectangle(img, (0, 0), (400, 190), (96, 96, 96), cv2.FILLED)
        img[mask] = cv2.addWeighted(img_copy, alpha, shapes, 1 - alpha, gamma=0)[mask]

        y_pos = 30
        if num_detections_camera > 0:
            cv2.putText(
                img,
                "Camera detections: " + str(num_detections_camera),
                (10, y_pos),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (96, 255, 96),
                1,
            )
            y_pos += 50
        # plot number of LiDAR detections
        if num_detections_lidar > 0:
            cv2.putText(
                img,
                "LiDAR detections:   " + str(num_detections_lidar),
                (10, y_pos),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (235, 192, 52),
                1,
            )
            y_pos += 50
        if num_detections_fused > 0:
            # plot number of fused detections
            cv2.putText(
                img,
                "Fused detections:   " + str(num_detections_fused),
                (10, y_pos),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (96, 96, 255),
                1,
            )
            y_pos += 50
        if num_detections_lidar > 0 and num_detections_camera > 0:
            # plot number of total detections
            cv2.putText(
                img,
                "Total detections:    " + str(num_detections_camera + num_detections_lidar - num_detections_fused),
                (10, y_pos),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 255, 255),
                1,
            )

    def draw_line(self, img, start_point, end_point, color):
        cv2.line(img, start_point, end_point, color, 2)

    def draw_3d_box_by_keypoints(self, img, box_3d_label, color, image_width, image_height):
        # draw bottom 4 lines
        start_point = (int(box_3d_label["box3d_projected"]["bottom_left_front"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["bottom_left_front"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["bottom_left_back"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["bottom_left_back"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["bottom_left_back"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["bottom_left_back"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["bottom_right_back"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["bottom_right_back"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["bottom_right_back"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["bottom_right_back"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["bottom_right_front"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["bottom_right_front"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["bottom_right_front"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["bottom_right_front"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["bottom_left_front"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["bottom_left_front"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        # draw top 4 lines
        start_point = (int(box_3d_label["box3d_projected"]["top_left_front"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["top_left_front"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["top_left_back"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["top_left_back"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["top_left_back"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["top_left_back"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["top_right_back"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["top_right_back"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["top_right_back"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["top_right_back"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["top_right_front"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["top_right_front"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["top_right_front"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["top_right_front"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["top_left_front"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["top_left_front"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        # draw 4 vertical lines
        start_point = (int(box_3d_label["box3d_projected"]["bottom_left_front"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["bottom_left_front"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["top_left_front"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["top_left_front"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["bottom_left_back"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["bottom_left_back"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["top_left_back"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["top_left_back"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["bottom_right_back"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["bottom_right_back"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["top_right_back"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["top_right_back"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)

        start_point = (int(box_3d_label["box3d_projected"]["bottom_right_front"][0] * image_width),
                       int(box_3d_label["box3d_projected"]["bottom_right_front"][1] * image_height))
        end_point = (int(box_3d_label["box3d_projected"]["top_right_front"][0] * image_width),
                     int(box_3d_label["box3d_projected"]["top_right_front"][1] * image_height))
        self.draw_line(img, start_point, end_point, color)
