import argparse
import glob
import json
import math
import os
import sys
from random import randint

import cv2
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter
from src.utils.detection import Detection
from src.utils.utils import id_to_class_name_mapping, class_name_to_id_mapping, get_2d_corner_points, \
    providentia_to_tum_traffic_category_mapping

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../src", ".."))
from matplotlib import pyplot as plt, cm
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R
import matplotlib.patheffects as pe
import numpy as np
from src.utils.vis_utils import VisualizationUtils
import src.map.hd_map as hdmap


def get_class_color_by_name(class_name, dataset_classes):
    if dataset_classes == "tum_traffic":
        class_color = id_to_class_name_mapping[str(class_name_to_id_mapping[class_name])]["color_rgb_normalized"]
    elif dataset_classes == "providentia":
        class_color = id_to_class_name_mapping[
            str(class_name_to_id_mapping[providentia_to_tum_traffic_category_mapping[class_name]])][
            "color_rgb_normalized"]
    else:
        raise ValueError("Unknown dataset type")
    return class_color


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_boxes",
        type=str,
        help="Path to 3D boxes",
        default="",
    )
    arg_parser.add_argument(
        "--plot_legend",
        action="store_true",
        help="Plot legend",
    )
    arg_parser.add_argument(
        "--plot_traffic_participants",
        action="store_true",
        help="Plot legend",
    )
    arg_parser.add_argument(
        "--viz_color_mode",
        type=str,
        choices=["by_category", "by_track_id"],
        default="by_category",
        help="Visualization color mode",
    )

    arg_parser.add_argument(
        "--output_folder_path_statistic_plots",
        type=str,
        help="output folder path to statistics plots",
        default="",
    )
    arg_parser.add_argument(
        "--output_folder_path_heatmap",
        type=str,
        help="output file path to heatmap",
        default="",
    )
    arg_parser.add_argument(
        "--dataset_classes",
        type=str,
        help="dataset classes",
        default="tum_traffic")
    arg_parser.add_argument(
        "--sensor_station_id",
        type=str,
        help="sensor station id. Examples: s040, s050, s060, s070, s080, s090, s110",
        default="s110")
    arg_parser.add_argument(
        "--draw_box",
        action="store_true",
        help="Draw box",
    )
    arg_parser.add_argument(
        "--draw_arrow",
        action="store_true",
        help="Draw arrow",
    )
    arg_parser.add_argument(
        "--accumulate_tracks",
        action="store_true",
        help="Accumulate tracks",
    )
    arg_parser.add_argument(
        "--font_size",
        type=int,
        help="Font size",
        default=10,
    )
    args = arg_parser.parse_args()
    input_folder_path_boxes = args.input_folder_path_boxes
    dataset_classes = args.dataset_classes
    sensor_station_id = args.sensor_station_id
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots
    draw_box = args.draw_box
    draw_arrow = args.draw_arrow
    accumulate_tracks = args.accumulate_tracks
    font_size = args.font_size

    # create output folder if not exists
    if not os.path.exists(output_folder_path_statistic_plots):
        os.makedirs(output_folder_path_statistic_plots)

    utils = VisualizationUtils()

    if sensor_station_id == "s110":
        use_s110_base_frame = True
    else:
        use_s110_base_frame = False

    bev_fig = None
    bev_ax = None
    boxes_list = {}
    # road -> common road frame
    # s110_base -> s110 base frame
    if sensor_station_id == "s110":
        # s110 base frame
        lane_sections = hdmap.load_map_for_local_frame("s110_base")
    elif sensor_station_id == "s040" or sensor_station_id == "s050":
        lane_sections = hdmap.load_map_for_local_frame("road")
    else:
        raise ValueError("Unknown sensor station id")

    # filter lane section to 200 m x 200 m region around s110 base
    lane_sections_filtered = []
    for lane_section in lane_sections:
        # for 110_base:
        if sensor_station_id == "s110":
            lane_section = lane_section.crop_to_area(min_pos=np.array([-160, -100]), max_pos=np.array([160, 120]))
            # lanes_ax.set_xlim(-160, 160)
            # lanes_ax.set_ylim(-80, 120)
        elif sensor_station_id == "s040" or sensor_station_id == "s050":
            lane_section = lane_section.crop_to_area(min_pos=np.array([-400, -100]), max_pos=np.array([600, 100]))
        else:
            raise ValueError("Unknown sensor station id")
        if lane_section:
            lane_sections_filtered.append(lane_section)

    # TUM Traffic Dataset
    if dataset_classes == "tum_traffic":
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
    elif dataset_classes == "providentia":
        # Providentia classes
        classes = ["PRE_TRACK",
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
    else:
        raise ValueError("Unknown dataset type")
    classes_valid_set = set()
    valid_ids = set()

    input_file_paths_boxes = sorted(glob.glob(os.path.join(input_folder_path_boxes, "*.json")))
    # raise ValueError("Unknown dataset type")
    for file_path in input_file_paths_boxes:
        file_name = os.path.basename(file_path)
        json_data = json.load(open(file_path))
        for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
            num_labeled_objects = len(frame_obj["objects"].keys())
            for object_track_id, object_json in frame_obj["objects"].items():
                object_class = object_json["object_data"]["type"]
                # NOTE: sometimes the dtwin has MOTORCYCLE as class name (coming from YOLOv7), sometimes BIKE (coming from the fusion result)
                if dataset_classes == "providentia" and object_class == "MOTORCYCLE":
                    object_class = "BIKE"

                classes_valid_set.add(object_class)
                valid_ids.add(classes.index(object_class))

                if "cuboid" in object_json["object_data"]:
                    cuboid = object_json["object_data"]["cuboid"]["val"]
                    location = cuboid[0:3]
                    quaternion = np.asarray(cuboid[3:7])
                    roll, pitch, yaw = R.from_quat(quaternion).as_euler("xyz", degrees=False)
                    track_history_attribute = VisualizationUtils.get_attribute_by_name(
                        object_json["object_data"]["cuboid"]["attributes"]["vec"], "track_history"
                    )
                    if boxes_list.get(file_name) is None:
                        boxes_list[file_name] = []
                    boxes_list[file_name].append(
                        Detection(
                            location=location,
                            dimensions=(cuboid[7], cuboid[8], cuboid[9]),
                            yaw=yaw,
                            category=object_class,
                            pos_history=track_history_attribute["val"],
                            uuid=object_track_id,
                        )
                    )

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    # bev_fig, bev_ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    # plt.subplots_adjust(left=0.1, right=1.0, top=0.98, bottom=0.12)
    # bev_ax.set_aspect("equal")
    # bev_ax.set_xlim(-50, 50)
    # bev_ax.set_ylim(0, 100)

    # # plot all lane sections from hd map
    # TODO: temp plot
    # bev_fig, bev_ax = plt.subplots(figsize=(4, 2.5), dpi=300)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
    # bev_ax.set_xlim(-400, 600)
    # bev_ax.set_ylim(-30, 30)
    # for lane_section in lane_sections_filtered:
    #     for lane in lane_section.lanes:
    #         bev_ax.plot(lane[:, 0], lane[:, 1], color=(0.3, 0.3, 0.3), linewidth=1.0, zorder=0)
    # bev_fig.savefig(
    #     os.path.join(args.output_folder_path_statistic_plots,
    #                  "lanes_0.jpg")
    # )

    # remove not valid classes
    classes_valid_list = list(classes_valid_set)
    for class_name in classes.copy():
        if class_name not in classes_valid_list:
            classes.remove(class_name)

    # set legend for plot using class names
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=get_class_color_by_name(class_name, dataset_classes),
            # color="black",
            lw=3,
            linewidth=3,
            path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()],
            markersize=5,
            markerfacecolor=(23 / 255, 87 / 255, 217 / 255, 1.0),
            markeredgecolor=(23 / 255, 87 / 255, 217 / 255, 1.0),
            markeredgewidth=1,

            label=class_name if class_name != "EMERGENCY_VEHICLE" else "EMERGENCY",
        )
        for class_name in classes
    ]

    # # plot legend with black edge color
    # bev_ax.legend(
    #     handles=legend_elements,
    #     loc="upper right",
    #     bbox_to_anchor=(1.0, 1.0),
    #     fontsize=font_size,
    #     frameon=True,
    #     edgecolor="black",
    # )
    # bev_ax.set_xlabel("Longitude [m]", fontsize=font_size)
    # bev_ax.set_ylabel("Latitude [m]", fontsize=font_size)
    # bev_ax.tick_params(axis="both", which="major", labelsize=font_size)
    # bev_ax.tick_params(axis="both", which="minor", labelsize=font_size-2)

    transformation_lidar_south_to_base = np.array(
        [
            [0.21479485, -0.9761028, 0.03296187, -15.87257873],
            [0.97627128, 0.21553835, 0.02091894, 2.30019086],
            [-0.02752358, 0.02768645, 0.99923767, 7.48077521],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # iterate all boxes
    frame_idx = 0
    points_3d = []
    if accumulate_tracks:
        # for s110
        if sensor_station_id == "s110":
            bev_fig, bev_ax = plt.subplots(figsize=(5, 4.5), dpi=300)
            plt.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.12)
        elif sensor_station_id == "s040" or sensor_station_id == "s050":
            bev_fig, bev_ax = plt.subplots(figsize=(12, 5.0), dpi=300)
            plt.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.12)

        for lane_section in lane_sections_filtered:
            for lane in lane_section.lanes:
                color = (0.9, 0.9, 0.9)  # light gray (for dark background)
                # color = (0.3, 0.3, 0.3) # dark gray (for white background)
                bev_ax.plot(lane[:, 0], lane[:, 1], color=color, linewidth=1.0, zorder=0)

        bev_ax.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=font_size,
            frameon=True,
            edgecolor="black",
        )
        # make black border around legend items

        bev_ax.tick_params(axis="both", which="major", labelsize=font_size)
        bev_ax.tick_params(axis="both", which="minor", labelsize=font_size-2)
        if sensor_station_id == "s110":
            bev_ax.xaxis.set_major_locator(MultipleLocator(40))
            bev_ax.set_xlabel("Longitude [m]", fontsize=font_size)
            bev_ax.set_ylabel("Latitude [m]", fontsize=font_size)
        elif sensor_station_id == "s040" or sensor_station_id == "s050":
            bev_ax.xaxis.set_major_locator(MultipleLocator(50))
            bev_ax.set_xlabel(r'Longitude [m]~~~~~~~~~~$\leftarrow$ North-South $\rightarrow$', fontsize=font_size)
            bev_ax.set_ylabel(r'Latitude [m]~~~~~~~~~~$\leftarrow$ West-East $\rightarrow$', fontsize=font_size)



    for label_file_name, boxes in boxes_list.items():
        if not accumulate_tracks:
            #bev_fig, bev_ax = plt.subplots(figsize=(5, 4.5), dpi=300)
            #plt.subplots_adjust(left=0.1, right=1.0, top=0.98, bottom=0.0)
            # create 1920x1200 figure (4:2.5 aspect ratio)
            bev_fig, bev_ax = plt.subplots(figsize=(4, 2.5), dpi=480)
            if sensor_station_id == "s110":
                plt.subplots_adjust(left=0.1, right=1.0, top=0.97, bottom=0.18)
            elif sensor_station_id == "s040" or sensor_station_id == "s050":
                plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.18)

        # bev_fig, bev_ax = plt.subplots(figsize=(4, 2.5), dpi=480)
        # set background color to dark gray
        bev_ax.set_facecolor((0.3, 0.3, 0.3))
        if sensor_station_id == "s110":
            # for s110
            #plt.subplots_adjust(left=0.16, right=0.99, top=0.95, bottom=0.22)
            # bev_ax.set_xlim(-80, 80)
            # bev_ax.set_ylim(-60, 100)

            bev_ax.set_xlim(-100, 60)
            bev_ax.set_ylim(-30, 70)
        elif sensor_station_id == "s040" or sensor_station_id == "s050":
            # for s40/s50
            #plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
            # for s40/s50
            bev_ax.set_xlim(-350, 500)
            bev_ax.set_ylim(-30, 50)
        if sensor_station_id == "s110":
            bev_ax.set_aspect("equal")
        # remove white space around the image
        #plt.margins(0, 0)


        # plot all lane sections from hd map
        if not accumulate_tracks:
            for lane_section in lane_sections_filtered:
                for lane in lane_section.lanes:
                    color = (0.9, 0.9, 0.9) # light gray (for dark background)
                    # color = (0.3, 0.3, 0.3) # dark gray (for white background)
                    bev_ax.plot(lane[:, 0], lane[:, 1], color=color, linewidth=1.0, zorder=0)

        if args.plot_legend and not accumulate_tracks:
            # plot legend with black edge color
            bev_ax.legend(
                handles=legend_elements,
                loc="upper right",
                bbox_to_anchor=(1.0, 1.0),
                fontsize=font_size-5,
                frameon=True,
                edgecolor="black",
            )
            # make black border around legend items
            bev_ax.set_xlabel("Longitude [m]", fontsize=font_size)
            bev_ax.set_ylabel("Latitude [m]", fontsize=font_size)
            bev_ax.tick_params(axis="both", which="major", labelsize=font_size)
            bev_ax.tick_params(axis="both", which="minor", labelsize=font_size-2)
            if sensor_station_id == "s110":
                bev_ax.xaxis.set_major_locator(MultipleLocator(20))
                # set x ticks distance to 20
            elif sensor_station_id == "s040" or sensor_station_id == "s050":
                bev_ax.xaxis.set_major_locator(MultipleLocator(100))

        for box_idx, box in enumerate(boxes):
            if dataset_classes == "tum_traffic":
                class_id = class_name_to_id_mapping[box.category.upper()]
            elif dataset_classes == "providentia":
                class_id = class_name_to_id_mapping[providentia_to_tum_traffic_category_mapping[box.category.upper()]]
            else:
                raise ValueError("Unknown dataset type")
            if args.viz_color_mode == "by_category":
                class_color_rgb_normalized = id_to_class_name_mapping[str(class_id)]["color_rgb_normalized"]
            elif args.viz_color_mode == "by_track_id":
                # get a random color for each track id
                if box.uuid not in utils.track_id_to_color_mapping:
                    random_number = randint(0, utils.num_colors - 1)
                    random_color_rgb_normalized = utils.random_colors[random_number]
                    random_color_rgb = tuple(
                        [int(x * 255) for x in random_color_rgb_normalized]
                    )
                    utils.track_id_to_color_mapping[box.uuid] = random_color_rgb
                class_color_rgb = utils.track_id_to_color_mapping[box.uuid]
                class_color_rgb_normalized = tuple(
                    [x / 255 for x in class_color_rgb]
                )
            box_rotation = box.yaw  # in radians
            rotation_matrix = np.array(
                [
                    [math.cos(box_rotation), -math.sin(box_rotation), 0],
                    [math.sin(box_rotation), math.cos(box_rotation), 0],
                    [0, 0, 1],
                ])
            # location in lidar south
            box_position = np.array([box.location[0], box.location[1], box.location[2], 1])
            # location in s110_base
            if use_s110_base_frame:
                box_position = np.matmul(transformation_lidar_south_to_base, box_position.T).T
                rotation_lidar_south_to_s110_base = 78
                rotation_lidar_south_to_s110_base_in_rad = rotation_lidar_south_to_s110_base * math.pi / 180
                # fill rectangle with color
                yaw_rotation = box.yaw + rotation_lidar_south_to_s110_base_in_rad
            else:
                yaw_rotation = box.yaw

            points_3d.append(box_position)
            if draw_box:
                px, py = get_2d_corner_points(
                    box_position[0], box_position[1], box.dimensions[0], box.dimensions[1],
                    yaw_rotation
                )
                bev_ax.fill(px, py, color=class_color_rgb_normalized, alpha=0.5, zorder=2)
                bev_ax.plot(px, py, color="black", linewidth=1.0, zorder=3)
            if draw_arrow:
                bev_ax.arrow(
                    box_position[0],
                    box_position[1],
                    box.dimensions[0] * 0.6 * np.cos(
                        (yaw_rotation) % (2 * math.pi)),
                    box.dimensions[0] * 0.6 * np.sin(
                        (yaw_rotation) % (2 * math.pi)),
                    head_width=box.dimensions[1] * 0.3,
                    color="k",
                )

            # plot track
            if len(box.pos_history) > 0:


                # set z-order to 0.5 to make sure that the track is plotted behind the rectangle
                # (otherwise the rectangle would be hidden by the track)
                locations = np.reshape(box.pos_history, (-1, 3))

                # add current location to history at the beginning
                locations = np.vstack((box_position[0:3], locations))



                if use_s110_base_frame:
                    transformation_lidar_to_base = None
                    rotation_lidar_to_base = None
                    if "s110_lidar_ouster_south" in label_file_name:
                        # rotation_lidar_south_to_base = 85.5
                        rotation_lidar_south_to_base = 0.0
                        rotation_lidar_to_base = rotation_lidar_south_to_base
                        # transform from lidar south coordinate frame to s110 base coordinate frame
                        y_offset = 1.3
                        transformation_lidar_to_base = transformation_lidar_south_to_base
                    elif "s110_lidar_ouster_north" in label_file_name:
                        rotation_lidar_north_to_base = 0.0
                        rotation_lidar_to_base = rotation_lidar_north_to_base
                        x_offset = -1.0
                        y_offset = -2.0
                        transformation_lidar_north_to_base = np.array(
                            [
                                [-0.064419, -0.997922, 0.00169282, -2.08748],
                                [0.997875, -0.0644324, -0.00969147, 0.226579],
                                [0.0097804, 0.0010649, 0.999952, 8.29723],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                        transformation_lidar_to_base = transformation_lidar_north_to_base
                    else:
                        # label in camera or common road frame -> no transformation needed
                        transformation_lidar_to_base = np.eye(4)


                    locations = np.hstack((locations, np.ones((locations.shape[0], 1))))
                    locations_transformed = np.matmul(transformation_lidar_to_base, locations.T).T

                    # rotate locations around z-axis
                    locations_rotated = np.zeros_like(locations_transformed)
                    locations_rotated[:, 0] = locations_transformed[:, 0] * np.cos(
                        rotation_lidar_to_base * np.pi / 180
                    ) - locations_transformed[:, 1] * np.sin(rotation_lidar_to_base * np.pi / 180)
                    locations_rotated[:, 1] = locations_transformed[:, 0] * np.sin(
                        rotation_lidar_to_base * np.pi / 180
                    ) + locations_transformed[:, 1] * np.cos(rotation_lidar_to_base * np.pi / 180)
                else:
                    locations_rotated = locations

                bev_ax.plot(
                    np.array(locations_rotated)[:, 0],
                    np.array(locations_rotated)[:, 1],
                    color=class_color_rgb_normalized,
                    zorder=1,
                    linewidth=1.0,
                )
            else:
                print(f"Warning: No history for {box.id} {box.category}", flush=True)

        # draw heat map
        if args.output_folder_path_heatmap != "":
            sigma = 8
            points_3d_arr = np.array(points_3d)
            points_3d_arr = points_3d_arr.astype(int)
            heatmap, xedges, yedges = np.histogram2d(points_3d_arr[:, 0], points_3d_arr[:, 1], bins=400,
                                                     range=[[-100, 100], [0, 100]])
            heatmap = gaussian_filter(heatmap, sigma=sigma)
            extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
            heatmap_img = heatmap.T
            # turn y values upside down
            heatmap_img = np.flipud(heatmap_img)
            heatmap_img = cv2.resize(heatmap_img, (1920, 1200), interpolation=cv2.INTER_CUBIC)
            heatmap_img = heatmap_img / np.max(heatmap_img)
            heatmap_img = heatmap_img * 255
            heatmap_img = heatmap_img.astype(np.uint8)
            heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.output_folder_path_heatmap, label_file_name.replace(".json", ".jpg")),
                        heatmap_img)
        if not accumulate_tracks:
            bev_fig.savefig(
                os.path.join(args.output_folder_path_statistic_plots,
                             str(frame_idx).zfill(9) + "_" + str(
                                 label_file_name.replace(".json", "")) + ".jpg", )
            )
            bev_fig.clf()
            plt.close(bev_fig)
        frame_idx += 1

    # draw x-axis arrow in red and y-axis arrow in green at origin
    if sensor_station_id == "s110":
        bev_ax.arrow(0, 0, 50, 0, head_width=2, head_length=2, fc="r", ec="r")
        bev_ax.arrow(0, 0, 0, 50, head_width=2, head_length=2, fc="g", ec="g")
    elif sensor_station_id == "s040" or sensor_station_id == "s050":
        bev_ax.arrow(0, 0, 100, 0, head_width=5, head_length=15, fc="r", ec="r")
        bev_ax.arrow(0, 0, 0, 30, head_width=10, head_length=5, fc=(0.478, 1.0, 0.184), ec=(0.478, 1.0, 0.184))

    print(f"Saving BEV plot to {args.output_folder_path_statistic_plots}", flush=True)
    bev_fig.savefig(os.path.join(output_folder_path_statistic_plots, "bev_plot_all_drives.pdf"))
    bev_fig.clf()
    plt.close(bev_fig)


    # merge heatmap_img and lanes_img using addWeighted
    if args.output_folder_path_heatmap != "":
        lanes_fig, lanes_ax = plt.subplots(figsize=(4, 2.5), dpi=480)
        # for s110
        if sensor_station_id == "s110":
            lanes_ax.set_xlim(-80, 80)
            lanes_ax.set_ylim(-60, 100)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        elif sensor_station_id == "s040" or sensor_station_id == "s050":
            # for s40/s50
            lanes_ax.set_xlim(-400, 600)
            lanes_ax.set_ylim(-30, 30)
            plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
        else:
            raise ValueError("Unknown sensor station ID.")
        # remove all white space
        lanes_ax.axes.get_yaxis().set_visible(False)
        lanes_ax.axes.get_xaxis().set_visible(False)
        lanes_ax.set_aspect("equal")
        for lane_section in lane_sections_filtered:
            for lane in lane_section.lanes:
                lanes_ax.plot(lane[:, 0], lane[:, 1], color=(0.3, 0.3, 0.3), linewidth=1.0, zorder=0)
        # lanes_fig.savefig(os.path.join(args.output_folder_path_statistic_plots, "lanes.jpg"))
        # redraw the canvas
        lanes_fig.canvas.draw()
        # convert canvas to image
        lanes_img = np.fromstring(lanes_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        lanes_img = lanes_img.reshape(lanes_fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        lanes_img = cv2.cvtColor(lanes_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_folder_path_statistic_plots, "lanes.jpg"), lanes_img)
        lanes_with_heatmap = cv2.addWeighted(heatmap_img, 0.5, lanes_img, 0.5, 0)
        cv2.imwrite(os.path.join(args.output_folder_path_heatmap, label_file_name.replace(".json", "_heatmap.jpg")),
                    lanes_with_heatmap)
    plt.close()
    plt.clf()
