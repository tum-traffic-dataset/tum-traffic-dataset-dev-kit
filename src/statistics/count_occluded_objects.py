import argparse
import copy
import os
import json
import numpy as np
from matplotlib import pyplot as plt

from src.utils.vis_utils import VisualizationUtils

if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_labels_train",
        type=str,
        help="Path to train labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_val",
        type=str,
        help="Path to val labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sampled",
        type=str,
        help="Path to test sampled labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sequence",
        type=str,
        help="Path to test sequence labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_south",
        type=str,
        help="Path to r02_s01 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_north",
        type=str,
        help="Path to r02_s01 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_south",
        type=str,
        help="Path to r02_s02 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_north",
        type=str,
        help="Path to r02_s02 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_south",
        type=str,
        help="Path to r02_s03 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_north",
        type=str,
        help="Path to r02_s03 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_south",
        type=str,
        help="Path to r02_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_north",
        type=str,
        help="Path to r02_s04 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--font_size",
        type=int,
        help="Font size for the plots",
        default=9,
    )
    arg_parser.add_argument(
        "--output_folder_path_statistic_plots",
        type=str,
        help="Path to output folder for plots",
        default="output",
    )
    args = arg_parser.parse_args()
    font_size = args.font_size
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots
    if not os.path.exists(output_folder_path_statistic_plots):
        os.makedirs(output_folder_path_statistic_plots)

    input_folder_paths_all = []
    input_folder_paths_all = []
    if args.input_folder_path_labels_train:
        input_folder_paths_all.append(args.input_folder_path_labels_train)
    if args.input_folder_path_labels_val:
        input_folder_paths_all.append(args.input_folder_path_labels_val)
    if args.input_folder_path_labels_test_sampled:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sampled)
    if args.input_folder_path_labels_test_sequence:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sequence)

    if args.input_folder_path_labels_sequence_s01_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_south)
    if args.input_folder_path_labels_sequence_s01_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_north)
    if args.input_folder_path_labels_sequence_s02_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_south)
    if args.input_folder_path_labels_sequence_s02_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_north)
    if args.input_folder_path_labels_sequence_s03_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_south)
    if args.input_folder_path_labels_sequence_s03_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_north)
    if args.input_folder_path_labels_sequence_s04_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_south)
    if args.input_folder_path_labels_sequence_s04_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_north)

    classes = {
        "CAR": [],
        "TRUCK": [],
        "TRAILER": [],
        "VAN": [],
        "MOTORCYCLE": [],
        "BUS": [],
        "PEDESTRIAN": [],
        "BICYCLE": [],
        "EMERGENCY_VEH": [],
        "OTHER": [],
    }
    occlusion_levels = {
        "NOT_OCCLUDED": 0,
        "PARTIALLY_OCCLUDED": 0,
        "MOSTLY_OCCLUDED": 0,
        "UNKNOWN": 0,
    }
    occlusion_levels_list = ["NOT_OCCLUDED", "PARTIALLY_OCCLUDED", "MOSTLY_OCCLUDED"]
    classes_and_occlusion_level = {
        "CAR": copy.deepcopy(occlusion_levels),
        "TRUCK": copy.deepcopy(occlusion_levels),
        "TRAILER": copy.deepcopy(occlusion_levels),
        "VAN": copy.deepcopy(occlusion_levels),
        "MOTORCYCLE": copy.deepcopy(occlusion_levels),
        "BUS": copy.deepcopy(occlusion_levels),
        "PEDESTRIAN": copy.deepcopy(occlusion_levels),
        "BICYCLE": copy.deepcopy(occlusion_levels),
        "EMERGENCY_VEHICLE": copy.deepcopy(occlusion_levels),
        "OTHER": copy.deepcopy(occlusion_levels),
    }

    class_colors = [
        (0, 0.8, 0.96, 0.3),
        (0.25, 0.91, 0.72, 0.3),
        (0.35, 1, 0.49, 0.3),
        (0.92, 0.81, 0.21, 0.3),
        (0.72, 0.64, 0.33, 0.3),
        (0.85, 0.54, 0.52, 0.3),
        (0.91, 0.46, 0.97, 0.3),
        (0.69, 0.55, 1, 0.3),
        (0.4, 0.42, 0.98, 0.3),
        (0.78, 0.78, 0.78, 0.3),
    ]

    num_not_occluded = 0
    num_partially_occluded = 0
    num_mostly_occluded = 0
    total_objects = 0
    num_unknown = 0
    num_attributes = 0
    for input_folder_path in input_folder_paths_all:
        input_files_labels = sorted(os.listdir(input_folder_path))

        for label_file_name in input_files_labels:
            json_file = open(
                os.path.join(input_folder_path, label_file_name),
            )
            json_data = json.load(json_file)
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                for object_track_id, object_json in frame_obj["objects"].items():
                    object_data = object_json["object_data"]
                    if np.all(np.array(object_data["cuboid"]["val"]) == 0):
                        print("Object with all zeros")
                        continue
                    else:
                        total_objects += 1
                        if "attributes" in object_data["cuboid"]:
                            attribute = VisualizationUtils.get_attribute_by_name(
                                object_data["cuboid"]["attributes"]["text"], "occlusion_level"
                            )
                            if "text" in object_data["cuboid"]["attributes"]:
                                num_attributes += len(object_data["cuboid"]["attributes"]["text"])
                            if "num" in object_data["cuboid"]["attributes"]:
                                num_attributes += len(object_data["cuboid"]["attributes"]["num"])
                            if "boolean" in object_data["cuboid"]["attributes"]:
                                num_attributes += len(object_data["cuboid"]["attributes"]["boolean"])
                            if attribute is not None:
                                occlusion_level = attribute["val"]
                                if occlusion_level == "NOT_OCCLUDED":
                                    num_not_occluded += 1
                                    classes_and_occlusion_level[object_data["type"]][occlusion_level] = int(
                                        int(classes_and_occlusion_level[object_data["type"]][occlusion_level]) + 1
                                    )
                                elif occlusion_level == "PARTIALLY_OCCLUDED":
                                    num_partially_occluded += 1
                                    classes_and_occlusion_level[object_data["type"]][occlusion_level] = int(
                                        int(classes_and_occlusion_level[object_data["type"]][occlusion_level]) + 1
                                    )
                                elif occlusion_level == "MOSTLY_OCCLUDED":
                                    num_mostly_occluded += 1
                                    classes_and_occlusion_level[object_data["type"]][occlusion_level] = int(
                                        int(classes_and_occlusion_level[object_data["type"]][occlusion_level]) + 1
                                    )
                                elif occlusion_level == "":
                                    classes_and_occlusion_level[object_data["type"]]["UNKNOWN"] = int(
                                        int(classes_and_occlusion_level[object_data["type"]]["UNKNOWN"]) + 1
                                    )
                                    num_unknown += 1
                                else:
                                    print("Unknown occlusion level: " + occlusion_level)
                            else:
                                num_unknown += 1
                                print("No occlusion attribute")

    print("num_not_occluded:", str(num_not_occluded))
    print("num_partially_occluded:", str(num_partially_occluded))
    print("num_mostly_occluded:", str(num_mostly_occluded))
    print("unknown:", str(num_unknown))
    print("total_objects:", str(total_objects))
    print("num_attributes:", str(num_attributes))

    #############################
    # 1. Bar plot for occlusion levels
    #############################
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    #fig, ax = plt.subplots(figsize=(5, 4.5))
    #plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.32)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.41)

    class_names = classes.keys()
    occlusion_level_not_occluded = [
        occlusion_level_dict["NOT_OCCLUDED"] for occlusion_level_dict in classes_and_occlusion_level.values()
    ]
    occlusion_level_partially_occluded = [
        occlusion_level_dict["PARTIALLY_OCCLUDED"] for occlusion_level_dict in classes_and_occlusion_level.values()
    ]
    occlusion_level_mostly_occluded = [
        occlusion_level_dict["MOSTLY_OCCLUDED"] for occlusion_level_dict in classes_and_occlusion_level.values()
    ]
    # TODO: if before fusion, then comment out
    # occlusion_level_unknown = [
    #     occlusion_level_dict["UNKNOWN"] for occlusion_level_dict in classes_and_occlusion_level.values()
    # ]
    # add up (sum) all occlusion level arrays
    occlusion_level_all = np.array(
        [
            occlusion_level_mostly_occluded,
            occlusion_level_partially_occluded,
            occlusion_level_not_occluded,
            # occlusion_level_unknown,  # TODO: if before fusion, then comment out
        ]
    )
    occlusion_level_all = np.add(0, occlusion_level_all.sum(axis=0))
    plt.bar(
        class_names,
        occlusion_level_not_occluded,
        #color="black",
        color=class_colors,
        edgecolor="black",
        label="NOT_OCCLUDED",
    )
    plt.bar(
        class_names,
        occlusion_level_partially_occluded,
        # bottom=occlusion_level_not_occluded,
        color=class_colors,
        edgecolor="black",
        hatch="////",
        label="PARTIALLY_OCCLUDED",
    )
    plt.bar(
        class_names,
        occlusion_level_mostly_occluded,
        # bottom=np.add(occlusion_level_not_occluded, occlusion_level_partially_occluded),
        color=class_colors,
        edgecolor="black",
        hatch="xxxx",
        label="MOSTLY_OCCLUDED",
    )


    # TODO: if before fusion, then comment out
    # plt.bar(
    #     class_names,
    #     occlusion_level_unknown,
    #     bottom=np.add(
    #         occlusion_level_mostly_occluded, np.add(occlusion_level_not_occluded, occlusion_level_partially_occluded)
    #     ),
    #     color=class_colors,
    #     edgecolor="black",
    #     hatch="\\\\",
    #     label="UNKNOWN",
    # )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(r"\# 3D box labels", fontsize=font_size)
    plt.yscale("log")
    ax.set_ylim([1, 10000000])
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=font_size-2)
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color("white")
    leg.legendHandles[0].set_edgecolor("black")
    leg.legendHandles[1].set_color("white")
    leg.legendHandles[1].set_edgecolor("black")
    leg.legendHandles[2].set_color("white")
    leg.legendHandles[2].set_edgecolor("black")

    # TODO: if before fusion, then comment out
    # leg.legendHandles[3].set_color("white")
    # leg.legendHandles[3].set_edgecolor("black")

    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    # calculate average y value
    average_y = int(round(np.mean(occlusion_level_all)))

    plt.axhline(y=average_y, linewidth=1, color="r", linestyle="--", zorder=3)
    plt.text(4, average_y + 1000, str(average_y), color="r", fontweight="bold", fontsize=font_size)
    for i, num_labels in enumerate(occlusion_level_all):
        ax.text(i - 0.3, num_labels + num_labels / 4, str(num_labels), color="black", fontweight="bold", fontsize=font_size)
    # change the fontsize of ticks label
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.tick_params(axis="both", which="minor", labelsize=font_size-2)
    plt.text(-0.5, average_y + 500000, "Total 3D box labels: " + str(total_objects),
             color="black",
             fontweight="bold",
             fontsize=font_size)
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "bar_chart_occlusion_level.pdf"))

    #############################
    # 2. Pie chart for occlusion levels
    #############################
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.subplots_adjust(left=0, right=0.55, top=1.0, bottom=0.1)
    wedges, texts, autotexts = plt.pie(
        [num_not_occluded, num_partially_occluded, num_mostly_occluded],
        # labels=["NOT_OCCLUDED", "PARTIALLY_OCCLUDED", "MOSTLY_OCCLUDED", "UNKNOWN"],
        colors=[
            (136 / 255, 191 / 255, 118 / 255, 1.0),
            (61 / 255, 185 / 255, 154 / 255, 1.0),
            (23 / 255, 87 / 255, 217 / 255, 1.0),
            # (10 / 255, 172 / 255, 189 / 255, 1.0),
        ],
        wedgeprops={"edgecolor": "black", "linewidth": 1, "width": 0.6},
        startangle=-40,
        autopct="%1.1f%%",
        textprops=dict(color="w", weight="bold", fontsize=20),
    )

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        if i == 0:
            ang = 40
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(
            occlusion_levels_list[i],
            xy=(x, y),
            xytext=(1.35 * np.sign(x), 1.4 * y),
            horizontalalignment=horizontalalignment,
            **kw,
        )

    for i, t in enumerate(texts):
        t.set_color("white")
        t.set_fontsize(font_size-2)
        t.set_fontweight("bold")
        t.set_fontstyle("italic")

    plt.setp(autotexts, size=font_size, weight="bold")
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "pie_chart_occlusion_level.pdf"))
