import glob
import json
import os
import sys
from pathlib import Path
from random import randint

# add repository root to python path
from src.utils.detection import Detection
from src.utils.perspective import parse_perspective

sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2
import argparse
import open3d as o3d
import numpy as np

from src.utils.vis_utils import VisualizationUtils
from src.utils.utils import id_to_class_name_mapping, class_name_to_id_mapping

np.set_printoptions(suppress=True)
from scipy.spatial.transform import Rotation as R


# Example:
# python visualize_image_with_3d_boxes.py --input_folder_path_images images --input_folder_path_point_clouds point_clouds --input_folder_path_labels labels --camera_id s110_camera_basler_south1_8mm --lidar_id s110_lidar_ouster_south --use_detection_boxes_in_s110_base  --viz_mode [box2d,box3d,point_cloud,track_history] --output_folder_path_visualization visualization --file_path_calibration_data calib/vehicle_camera_basler_16mm.json


def process_boxes(img, box_data, use_two_colors, input_type, camera_id, lidar_id, perspective,
                  boxes_coordinate_system_origin, transformation_matrix_vehicle_lidar_to_infra_lidar,dataset_release):
    """
    :param img:                     Input image
    :param box_data:                JSON data of the boxes
    :param use_two_colors:          If True, use two colors for the boxes
    :param input_type:              Type of the input data (labels or detections)
    :param camera_id:               ID of the camera
    :param lidar_id:                ID of the lidar
    :param perspective:             Perspective object of the camera
    :param boxes_coordinate_system_origin:  Origin coordinate system of boxes. Possible values: [s110_base, s110_lidar_ouster_south, vehicle_lidar_robosense].
    :param transformation_matrix_vehicle_lidar_to_infra_lidar:   Transformation matrix from vehicle lidar to infra lidar
    :return:
    """
    polylines_list = []
    if "openlabel" in box_data:
        # detections = []
        num_detections_lidar = 0
        num_detections_camera = 0
        num_detections_fused = 0
        for frame_idx, frame_obj in box_data["openlabel"]["frames"].items():
            for box_idx, box in frame_obj["objects"].items():
                category = box["object_data"]["type"]

                # TODO: temporarily hard code sensor ID (until it is included into mask attributes)
                color_rgb = get_color(box_idx, category, input_type, camera_id, use_two_colors)
                # swap channels because opencv uses bgr
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                # extract mask
                if "mask" in viz_mode:
                    if "poly2d" in box["object_data"]:
                        masks = box["object_data"]["poly2d"]
                        # get full mask belonging to correct sensor ID
                        full_mask = None
                        for mask in masks:
                            # TODO: temp do not check camera_id
                            # if mask["name"] == "full_mask" and mask["attributes"]["text"][0]["val"] == camera_id:
                            if mask["name"] == "full_mask":
                                full_mask = mask
                                break
                        if full_mask is not None:
                            polygons = np.array(full_mask["val"]).reshape(-1, 2)
                            polygons = np.array([polygons], dtype=np.int32)
                            polylines_list.append(polygons)
                            # use fillConvexPoly to fill the mask
                            overlay = img.copy()
                            cv2.fillPoly(overlay, polygons, color_bgr)
                            alpha = 0.5
                            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                if "box2d" in viz_mode:
                    # extract 2D box
                    if "bbox" in box["object_data"]:
                        # get full bbox
                        full_bbox = None
                        if len(box["object_data"]["bbox"]) > 0:
                            if len(box["object_data"]["bbox"]) == 1:
                                full_bbox = box["object_data"]["bbox"][0]
                            else:
                                for bbox in box["object_data"]["bbox"]:
                                    if bbox["name"] == "full_bbox":
                                        full_bbox = bbox
                                        break
                            if full_bbox is not None:
                                # draw 2D box
                                x_center = int(full_bbox["val"][0])
                                y_center = int(full_bbox["val"][1])
                                width = int(full_bbox["val"][2])
                                height = int(full_bbox["val"][3])
                                cv2.rectangle(img, (int(x_center - width / 2), int(y_center - height / 2)),
                                              (int(x_center + width / 2), int(y_center + height / 2)), color_bgr, 2)
                                # plot banner
                                class_label = box["object_data"]["type"] + "_" + box["object_data"]["name"].split("_")[
                                                                                     1][:4]
                                utils.plot_banner(img, int(x_center - width / 2), int(y_center - height / 2),
                                                  int(x_center + width / 2), int(y_center + height / 2), color_bgr,
                                                  class_label)

                if "box3d_projected" in viz_mode:
                    utils.draw_pseudo_3d_box(img, box["object_data"], color_bgr, use_normalized_keypoints=False)

                if "cuboid" in box["object_data"]:
                    cuboid = box["object_data"]["cuboid"]["val"]
                    location = np.array([[cuboid[0]], [cuboid[1]], [cuboid[2]]])
                    if dataset_release == "r00":
                        # add half of the height to the z position
                        location[2] += cuboid[9] / 2
                    quaternion = [cuboid[3], cuboid[4], cuboid[5], cuboid[6]]
                    roll, pitch, yaw = R.from_quat(quaternion).as_euler("xyz", degrees=False)
                    pos_history = np.array([])
                    if "attributes" in box["object_data"]["cuboid"]:
                        attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["text"], "body_color"
                        )
                        if attribute is not None:
                            color = attribute["val"]
                        else:
                            color = ""
                        attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["num"], "num_points"
                        )
                        if attribute is not None:
                            num_points = int(attribute["val"])
                        else:
                            num_points = -1

                        attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["num"], "score"
                        )
                        if attribute is not None:
                            score = round(float(attribute["val"]), 2)
                        else:
                            score = -1

                        # extract sensor ID as attribute to each detected object
                        attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["text"], "sensor_id"
                        )
                        if attribute is not None:
                            sensor_id = attribute["val"]
                        else:
                            sensor_id = ""


                        if "vec" in box["object_data"]["cuboid"]["attributes"]:
                            attribute = VisualizationUtils.get_attribute_by_name(
                                box["object_data"]["cuboid"]["attributes"]["vec"], "track_history"
                            )
                            if attribute is not None:
                                track_history_list = attribute["val"]
                                pos_history = np.array(track_history_list).reshape(-1, 3)
                            else:
                                pos_history = np.array([])
                    else:
                        color = ""
                        num_points = -1
                        score = -1
                        pos_history = np.array([])

                    if boxes_coordinate_system_origin == "s110_base" and lidar_id == "vehicle_lidar_robosense":
                        # transform boxes from s110_base to s110_lidar_ouster_south lidar coordinate system
                        location, rotation = perspective.transform_from_s110_base_to_s110_lidar_ouster_south(location,
                                                                                                             roll,
                                                                                                             pitch, yaw)
                        roll = rotation[0]
                        pitch = rotation[1]
                        yaw = rotation[2]

                    if lidar_id == "vehicle_lidar_robosense":
                        # transform boxes from s110_lidar_ouster_south lidar coordinate system to vehicle_lidar_robosense coordinate system
                        # drive_07: 2, drive_15: 0
                        roll_offset_deg = 2
                        roll_offset_rad = np.deg2rad(roll_offset_deg)
                        yaw_deg = np.rad2deg(yaw)
                        if yaw_deg < 0:
                            yaw_deg = 360 + yaw_deg
                        if yaw_deg > 360:
                            yaw_deg = yaw_deg - 360
                        if yaw_deg >= 0 and yaw_deg <= 180:
                            roll = roll - roll_offset_rad
                        else:
                            roll = roll + roll_offset_rad

                        box_pose = np.eye(4)
                        box_pose[0:3, 3] = location[:, 0]
                        box_pose[0:3, 0:3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()
                        box_pose_transformed = np.linalg.inv(
                            transformation_matrix_vehicle_lidar_to_infra_lidar) @ box_pose
                        rotation_matrix_new = box_pose_transformed[0:3, 0:3]
                        yaw = R.from_matrix(box_pose_transformed[0:3, 0:3]).as_euler("xyz", degrees=False)[2]
                        quaternion_new = R.from_matrix(rotation_matrix_new).as_quat()
                        box["object_data"]["cuboid"]["val"][0] = box_pose_transformed[0, 3]
                        box["object_data"]["cuboid"]["val"][1] = box_pose_transformed[1, 3]
                        box["object_data"]["cuboid"]["val"][2] = box_pose_transformed[2, 3]
                        box["object_data"]["cuboid"]["val"][3] = quaternion_new[0]
                        box["object_data"]["cuboid"]["val"][4] = quaternion_new[1]
                        box["object_data"]["cuboid"]["val"][5] = quaternion_new[2]
                        box["object_data"]["cuboid"]["val"][6] = quaternion_new[3]
                        # TODO: transform all track history positions
                        if pos_history.size > 0:
                            pos_history_transformed = np.zeros_like(pos_history)
                            for i in range(pos_history.shape[0]):
                                position_transformed = np.linalg.inv(
                                    transformation_matrix_vehicle_lidar_to_infra_lidar) @ np.array(
                                    [pos_history[i, 0], pos_history[i, 1], pos_history[i, 2], 1])
                                pos_history_transformed[i, :] = position_transformed[0:3]
                            pos_history = pos_history_transformed[:, 0:3]

                    # TODO: temp set sensor ID for mono3d detections
                    # sensor_id = "s110_camera_basler_south2_8mm"

                    if "lidar" in sensor_id and not "camera" in sensor_id:
                        num_detections_lidar += 1
                    elif "camera" in sensor_id and not "lidar" in sensor_id:
                        num_detections_camera += 1
                    elif "camera" in sensor_id and "lidar" in sensor_id:
                        num_detections_lidar += 1
                        num_detections_camera += 1
                        num_detections_fused += 1

                    detection = Detection(
                        location=location,
                        dimensions=(cuboid[7], cuboid[8], cuboid[9]),
                        yaw=yaw,
                        category=category,
                        uuid=box_idx,
                        color=color,
                        score=score,
                        num_lidar_points=num_points,
                        pos_history=pos_history,
                    )
                    if "box3d" in viz_mode:
                        points_2d = utils.draw_3d_box(img, box, color_bgr, camera_id, lidar_id,
                                                      boxes_coordinate_system_origin,
                                                      perspective, dataset_release)
                        # plot banner
                        if "box2d" not in viz_mode and points_2d is not None:
                            if points_2d.shape[0] == 8 and points_2d.shape[1] == 2:
                                x_min = min(points_2d[:, 0])
                                y_min = min(points_2d[:, 1])
                                x_max = max(points_2d[:, 0])
                                y_max = max(points_2d[:, 1])
                                class_label = box["object_data"]["type"] + "_" + box["object_data"]["name"].split("_")[
                                                                                     1][:3]
                                # add speed in (km/h) to class_label
                                speed_attribute = VisualizationUtils.get_attribute_by_name(
                                    box["object_data"]["cuboid"]["attributes"]["num"], "speed"
                                )
                                if speed_attribute is not None:
                                    class_label += "_" + str(
                                        int(speed_attribute["val"] * 3.6)) + "kmh"
                                utils.plot_banner(img, x_min, y_min, x_max, y_max, color_bgr, class_label)
                    if "track_history" in viz_mode:
                        # draw track history
                        location_current = detection.location.flatten()
                        if boxes_coordinate_system_origin == "s110_base" or boxes_coordinate_system_origin == "common_road":
                            z_pos = 0
                        else:
                            z_pos = location_current[2] - detection.dimensions[2] / 2.0
                        # TODO: transform current location from infra lidar to vehicle lidar
                        location_transformed = np.linalg.inv(
                            transformation_matrix_vehicle_lidar_to_infra_lidar) @ np.array(
                            [location_current[0], location_current[1], z_pos, 1])
                        track_history_positions = [
                            np.array([
                                location_transformed[0],
                                location_transformed[1],
                                location_transformed[2],
                            ])
                        ]
                        for position3d in detection.pos_history:
                            if boxes_coordinate_system_origin == "common_road":
                                track_history_positions.append(np.array([position3d[0], position3d[1], 0]))
                            else:
                                track_history_positions.append(np.array([position3d[0], position3d[1], position3d[2]]))

                        projected_track_history = []
                        if len(track_history_positions) > 1:
                            projected_track_history = utils.project_3d_box_to_2d(
                                np.array(track_history_positions),
                                camera_id,
                                lidar_id,
                                boxes_coordinate_system_origin=boxes_coordinate_system_origin,
                            )
                            if projected_track_history is not None:
                                projected_track_history = np.array(projected_track_history).astype(int).T
                        # cv2.polylines(img, [projected_track_history], False, color_bgr, 8, cv2.LINE_AA)
                        if projected_track_history is not None and len(projected_track_history) > 1:
                            # iterate all segments and reduce line width from 8 to 0.01 based on the number of elements in projected_track_history
                            line_width_max = 8.0
                            line_width = line_width_max
                            for i in range(len(projected_track_history) - 1):
                                # calculate current line width
                                if int(line_width) > 0:
                                    cv2.line(
                                        img,
                                        tuple(projected_track_history[i]),
                                        tuple(projected_track_history[i + 1]),
                                        color_bgr,
                                        int(line_width),
                                        cv2.LINE_AA,
                                    )
                                delta = line_width_max / len(projected_track_history)
                                # normalize delta between 0 and 8
                                line_width -= delta

            if num_detections_lidar > 0 or num_detections_camera > 0 or num_detections_fused > 0:
                if "statistics" in viz_mode:
                    utils.plot_statistics(img, num_detections_lidar, num_detections_camera, num_detections_fused)

    # iterate all polylines and draw contours
    for polyline in polylines_list:
        # draw black contour
        cv2.polylines(img, [polyline], True, (0, 0, 0), 1)

    return img


def get_color(box_idx, category, input_type, sensor_id, use_two_colors):
    if use_two_colors and input_type == "detections":
        color_rgb = (245, 44, 71)  # red
    elif use_two_colors and input_type == "labels":
        color_rgb = (27, 250, 27)  # green
    else:
        if viz_color_mode == "by_category":
            # TODO: remove workaround after fixing labels
            if category != "SPECIAL_VEHICLE":
                color_rgb = id_to_class_name_mapping[str(class_name_to_id_mapping[category])][
                    "color_rgb"
                ]
            else:
                color_rgb = (102, 107, 250)
        elif viz_color_mode == "by_sensor_type":
            # TODO: parse sensor_id from attribute
            # sensor_id = "s110_lidar_ouster_south"
            if "lidar" in sensor_id and not "camera" in sensor_id:
                # set to light blue
                color_rgb = (52, 192, 235)
            elif "camera" in sensor_id and not "lidar" in sensor_id:
                # set to light green
                color_rgb = (96, 255, 96)
            elif "camera" in sensor_id and "lidar" in sensor_id:
                # set to red
                color_rgb = (255, 96, 96)
            else:
                print(
                    "Unknown sensor type. Please check sensor_id. Valid sensor types are: s110_lidar_ouster_south, s110_lidar_ouster_north, s110_camera_basler_south1_8mm, s110_camera_basler_south1_8mm. Exiting..."
                )
                sys.exit()
        elif viz_color_mode == "by_physical_color":
            # TODO: visualize by physical color
            print("Visualization by physical color is not yet implemented. Exiting...")
            sys.exit()
        elif viz_color_mode == "by_track_id":
            # get a random color for each track id
            if box_idx not in utils.track_id_to_color_mapping:
                random_number = randint(0, utils.num_colors - 1)
                random_color_rgb_normalized = utils.random_colors[random_number]
                random_color_rgb = tuple(
                    [int(x * 255) for x in random_color_rgb_normalized]
                )
                utils.track_id_to_color_mapping[box_idx] = random_color_rgb
            color_rgb = utils.track_id_to_color_mapping[box_idx]
        else:
            print(
                "Unknown color mode. Valid color modes are: by_category, by_sensor_type. Exiting..."
            )
            sys.exit()
    return color_rgb


def set_track_history(labels_json, label_file_paths, current_frame_idx, use_boxes_in_s110_base):
    for frame_idx, frame_obj in labels_json["openlabel"]["frames"].items():
        for uuid, box in frame_obj["objects"].items():
            if "cuboid" in box["object_data"]:
                track_history = []
                # iterate all previous frames and append historic position
                for idx in range(current_frame_idx - 1, -1, -1):
                    found_same_object = False
                    label_file_path_current = label_file_paths[idx]
                    labels_json_current = json.load(open(label_file_path_current))
                    for frame_idx_current, frame_obj_current in labels_json_current["openlabel"]["frames"].items():
                        for uuid_current, box_current in frame_obj_current["objects"].items():
                            if uuid_current == uuid:
                                found_same_object = True
                                if "cuboid" in box_current["object_data"]:
                                    cuboid_current = box_current["object_data"]["cuboid"]["val"]
                                    track_history.append(cuboid_current[0])
                                    track_history.append(cuboid_current[1])
                                    if use_boxes_in_s110_base:
                                        track_history.append(0)
                                    else:
                                        track_history.append(cuboid_current[2] - cuboid_current[9] / 2.0)
                    if not found_same_object:
                        break
                if "attributes" in box["object_data"]["cuboid"]:
                    track_history_attribute = None
                    if "vec" in box["object_data"]["cuboid"]["attributes"]:
                        track_history_attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["vec"], "track_history"
                        )
                    else:
                        box["object_data"]["cuboid"]["attributes"]["vec"] = []
                    if track_history_attribute is None:
                        # create track history attribute and add current location
                        track_history_attribute = {"name": "track_history", "val": track_history}
                        box["object_data"]["cuboid"]["attributes"]["vec"].append(track_history_attribute)
                    else:
                        # update track history attribute
                        track_history_attribute["val"] = track_history


def get_v2i_transformation_matrix(labels_json):
    for frame_id, frame_obj in labels_json["openlabel"]["frames"].items():
        if "frame_properties" in frame_obj:
            if "vehicle_lidar_robosense_to_s110_lidar_ouster_south" in frame_obj["frame_properties"]["transforms"]:
                transformation_matrix_vehicle_lidar_to_infra_lidar = np.array(
                    frame_obj["frame_properties"]["transforms"][
                        "vehicle_lidar_robosense_to_s110_lidar_ouster_south"][
                        "transform_src_to_dst"]["matrix4x4"])
                return transformation_matrix_vehicle_lidar_to_infra_lidar
            else:
                return None
    return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="VizLabel Argument Parser")
    argparser.add_argument(
        "--input_folder_path_images",
        default="images",
        help="Image folder path. Default: images",
    )
    argparser.add_argument(
        "--input_folder_path_point_clouds",
        default="",
        help="Point cloud folder path. Default: point_clouds",
    )
    argparser.add_argument(
        "--input_folder_path_labels",
        default=None,
        help="Input folder path to lidar labels in OpenLABEL format.",
    )
    argparser.add_argument(
        "--input_folder_path_detections",
        default=None,
        help="Input folder path to lidar detections in OpenLABEL format.",
    )
    argparser.add_argument(
        "--camera_id",
        default="",
        help="Camera id. Default: s110_camera_basler_south1_8mm",
    )
    argparser.add_argument(
        "--lidar_id",
        default="",
        help="Lidar id. Default: s110_lidar_ouster_south",
    )
    # add file path to calibration data
    argparser.add_argument(
        "--file_path_calibration_data",
        default="",
        help="File path to calibration data. Default: calib/calibration.json",
    )
    argparser.add_argument(
        "--labels_coordinate_system_origin",
        default="s110_lidar_ouster_south",
        help="Origin coordinate system of label boxes. Possible values are: [common_road, s110_base, s110_lidar_ouster_south, vehicle_lidar_robosense]. Default: s110_lidar_ouster_south",
    )
    argparser.add_argument(
        "--detections_coordinate_system_origin",
        default="s110_lidar_ouster_south",
        help="Origin coordinate system of detection boxes. Possible values are: [s110_base, s110_lidar_ouster_south, vehicle_lidar_robosense]. Default: s110_lidar_ouster_south",
    )
    argparser.add_argument(
        "--input_folder_path_v2i_transformation_matrices",
        default=None,
        help="input folder path to v2i transformation matrices",
    )
    argparser.add_argument(
        "--viz_mode",
        default="box3d",
        help="Visualization mode. Available modes are: [box2d, box3d_projected, box3d, point_cloud, track_history, mask]. Mode can be combined and separated by comma, e.g. box2d,box3d. Default: box3d",
    )
    argparser.add_argument(
        "--viz_color_mode",
        default="by_category",
        help="Visualization color mode. Available modes are: [by_category, by_physical_color, by_sensor_type]",
    )
    argparser.add_argument(
        "--output_folder_path_visualization",
        default="visualization",
        help="Output folder path to save visualization result to disk.",
    )
    argparser.add_argument(
        "--dataset_release",
        default="r00",
        help="Dataset release version. Default: r00. Possible values: [r00, r01, r02, r03, r04]",
    )
    args = argparser.parse_args()

    input_folder_path_images = args.input_folder_path_images
    input_folder_path_point_clouds = args.input_folder_path_point_clouds
    input_folder_path_labels = args.input_folder_path_labels
    input_folder_path_v2i_transformation_matrices = args.input_folder_path_v2i_transformation_matrices
    input_folder_path_detections = args.input_folder_path_detections
    camera_id = args.camera_id
    lidar_id = args.lidar_id
    labels_coordinate_system_origin = args.labels_coordinate_system_origin
    detections_coordinate_system_origin = args.detections_coordinate_system_origin
    viz_mode = args.viz_mode
    viz_color_mode = args.viz_color_mode
    output_folder_path_visualization = args.output_folder_path_visualization
    dataset_release = args.dataset_release

    if not os.path.exists(output_folder_path_visualization):
        Path(output_folder_path_visualization).mkdir(parents=True)

    utils = VisualizationUtils()

    camera_perspectives = {
        "s040_camera_basler_north_50mm": None,
        "s040_camera_basler_north_16mm": None,
        "s050_camera_basler_south_16mm": None,
        "s050_camera_basler_south_50mm": None,
        "s110_camera_basler_south1_8mm": None,
        "s110_camera_basler_south2_8mm": None,
        "s110_camera_basler_north_8mm": None,
        "s110_camera_basler_east_8mm": None,
        "vehicle_camera_basler_16mm": None,
    }

    types = ("*.jpg", "*.png")  # the tuple of image file types
    input_image_file_paths = []
    for files in types:
        input_image_file_paths.extend(sorted(glob.glob(input_folder_path_images + "/" + files)))
    print("Found {} images in {}".format(len(input_image_file_paths), input_folder_path_images))

    if input_folder_path_point_clouds != "":
        point_cloud_file_paths = sorted(glob.glob(input_folder_path_point_clouds + "/*.pcd"))
        print("Found {} point cloud files.".format(len(point_cloud_file_paths)))
    else:
        point_cloud_file_paths = [""] * len(input_image_file_paths)

    if input_folder_path_labels is not None:
        label_file_paths = sorted(glob.glob(input_folder_path_labels + "/*.json"))
        print("Found {} label files.".format(len(label_file_paths)))
    else:
        label_file_paths = [""] * len(input_image_file_paths)

    if input_folder_path_detections is not None:
        detection_file_paths = sorted(glob.glob(input_folder_path_detections + "/*.json"))
        print("Found {} detection files.".format(len(detection_file_paths)))
    else:
        detection_file_paths = [""] * len(input_image_file_paths)

    if input_folder_path_v2i_transformation_matrices is not None:
        v2i_transformation_matrices_file_paths = sorted(
            glob.glob(input_folder_path_v2i_transformation_matrices + "/*.json"))
        print("Found {} detection files.".format(len(v2i_transformation_matrices_file_paths)))
    else:
        v2i_transformation_matrices_file_paths = [""] * len(input_image_file_paths)

    parse_lidar_id = lidar_id == ""
    parse_camera_id = camera_id == ""

    current_frame_idx = 0
    for image_file_path, point_cloud_file_path, label_file_path, detection_file_path, v2i_transformation_matrices_file_path in zip(
            input_image_file_paths,
            point_cloud_file_paths,
            label_file_paths,
            detection_file_paths,
            v2i_transformation_matrices_file_paths
    ):
        print("image: ", image_file_path)
        # extract lidar id from label_file_path
        if parse_lidar_id:
            lidar_id = "_".join(Path(point_cloud_file_path).name.split(".")[0].split("_")[2:])
        if parse_camera_id:
            camera_id = "_".join(Path(image_file_path).name.split(".")[0].split("_")[2:])

        if camera_perspectives[camera_id] is None:
            camera_perspectives[camera_id] = parse_perspective(args.file_path_calibration_data)
            camera_perspectives[camera_id].initialize_matrices()

        img = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)

        transformation_matrix_vehicle_lidar_to_infra_lidar = None
        if label_file_path != "":
            labels_json = json.load(open(label_file_path))
            if camera_id == "vehicle_camera_basler_16mm":
                transformation_matrix_vehicle_lidar_to_infra_lidar = get_v2i_transformation_matrix(labels_json)
        else:
            labels_json = []

        if detection_file_path != "":
            detections_json = json.load(open(detection_file_path))
            transformation_matrix_vehicle_lidar_to_infra_lidar = get_v2i_transformation_matrix(detections_json)
        else:
            detections_json = []

        if "point_cloud" in viz_mode:
            if point_cloud_file_path != "":
                point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
                points_3d = np.asarray(point_cloud.points)

                if camera_id == "vehicle_camera_basler_16mm":
                    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
                    points_3d = np.matmul(np.linalg.inv(transformation_matrix_vehicle_lidar_to_infra_lidar),
                                          points_3d_homogeneous.T).T

                img = utils.project_point_cloud_to_image(img, points_3d, lidar_id=lidar_id,
                                                         camera_id=camera_id)
            else:
                print("No point cloud file found. Please set input_folder_path_point_clouds.")
                sys.exit(1)

        if v2i_transformation_matrices_file_path != "":
            v2i_transformation_matrices_json = json.load(open(v2i_transformation_matrices_file_path))
            for frame_id, frame_obj in v2i_transformation_matrices_json["openlabel"]["frames"].items():
                if "vehicle_lidar_robosense_to_s110_lidar_ouster_south" in frame_obj["frame_properties"]["transforms"]:
                    transformation_matrix_vehicle_lidar_to_infra_lidar = np.array(
                        frame_obj["frame_properties"]["transforms"][
                            "vehicle_lidar_robosense_to_s110_lidar_ouster_south"][
                            "transform_src_to_dst"]["matrix4x4"])
        else:
            transformation_matrix_vehicle_lidar_to_infra_lidar = np.eye(4)

        if "box2d" in viz_mode or "box3d" in viz_mode or "box3d_projected" in viz_mode or "track_history" in viz_mode or "mask" in viz_mode:
            use_two_colors = label_file_path != "" and detection_file_path != ""
            if label_file_path != "":
                # TODO: temp commented to improve speed, because track history is already calculated with set_track_history.py
                # if "track_history" in viz_mode:
                #     # set track history attribute
                #     set_track_history(
                #         labels_json,
                #         label_file_paths,
                #         current_frame_idx,
                #         use_boxes_in_s110_base=detections_coordinate_system_origin == "s110_base" or labels_coordinate_system_origin == "s110_base",
                #     )

                img = process_boxes(
                    img,
                    labels_json,
                    use_two_colors,
                    input_type="labels",
                    camera_id=camera_id,
                    lidar_id=lidar_id,
                    perspective=camera_perspectives[camera_id],
                    boxes_coordinate_system_origin=labels_coordinate_system_origin,
                    transformation_matrix_vehicle_lidar_to_infra_lidar=transformation_matrix_vehicle_lidar_to_infra_lidar,
                    dataset_release=dataset_release,
                )
            if detection_file_path != "":
                if "track_history" in viz_mode:
                    # set track history attribute
                    set_track_history(
                        detections_json,
                        detection_file_paths,
                        current_frame_idx,
                        use_boxes_in_s110_base=detections_coordinate_system_origin == "s110_base" or labels_coordinate_system_origin == "s110_base",
                    )
                img = process_boxes(
                    img,
                    detections_json,
                    use_two_colors,
                    input_type="detections",
                    camera_id=camera_id,
                    lidar_id=lidar_id,
                    perspective=camera_perspectives[camera_id],
                    boxes_coordinate_system_origin=detections_coordinate_system_origin,
                    transformation_matrix_vehicle_lidar_to_infra_lidar=transformation_matrix_vehicle_lidar_to_infra_lidar,
                    dataset_release=dataset_release,
                )
        if output_folder_path_visualization:
            cv2.imwrite(
                os.path.join(output_folder_path_visualization, Path(image_file_path).name.replace("png", "jpg")), img
            )
        else:
            cv2.imshow("image", img)
            cv2.waitKey()
        current_frame_idx = current_frame_idx + 1

    # destructor of VisualizationUtils must be called, otherwise open() can no longer be called since it is destroyed before it
    utils = None
