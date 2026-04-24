import argparse
import math
import os
import json
import numpy as np
import pandas

# This script calculates the average number of objects per frame
from matplotlib import pyplot as plt

from internal.src.statistics.plot_utils import PlotUtils

if __name__ == "__main__":
    class_colors = [
        (0, 0.8, 0.96, 1.0),
        (0.25, 0.91, 0.72, 1.0),
        (0.35, 1, 0.49, 1.0),
        (0.92, 0.81, 0.21, 1.0),
        (0.72, 0.64, 0.33, 1.0),
        (0.85, 0.54, 0.52, 1.0),
        (0.91, 0.46, 0.97, 1.0),
        (0.69, 0.55, 1, 1.0),
        (0.4, 0.42, 0.98, 1.0),
        (0.78, 0.78, 0.78, 1.0),
    ]
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
        help="Path to r01_s04 south lidar labels",
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
        "--output_folder_path_statistic_plots",
        type=str,
        help="Path to output folder",
        default="",
    )
    # add font size
    arg_parser.add_argument(
        "--font_size",
        type=int,
        help="Font size",
        default=9,
    )

    args = arg_parser.parse_args()
    font_size = args.font_size
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots

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

    num_frames = 0
    num_labeled_objects_all_frames_grouped = []
    interval_names = [
        r"[0-5)",
        r"[5-10)",
        r"[10-15)",
        r"[15-20)",
        r"[20-25)",
        r"[25-30)",
        r"[30-35)",
        r"[35-40)",
        r"[40-45)",
        r"[45-50)",
    ]
    # R00 Release
    # num_labeled_objects_all_frames = {
    #     "CAR": [],
    #     "TRUCK": [],
    #     "TRAILER": [],
    #     "VAN": [],
    #     "MOTORCYCLE": [],
    #     "BUS": [],
    #     "PEDESTRIAN": [],
    #     "BICYCLE": [],
    #     "SPECIAL_VEHICLE": [],
    #     "OTHER": [],
    # }
    # R01, R02, R03, R04 release
    num_labeled_objects_all_frames = {
        "CAR": [],
        "TRUCK": [],
        "TRAILER": [],
        "VAN": [],
        "MOTORCYCLE": [],
        "BUS": [],
        "PEDESTRIAN": [],
        "BICYCLE": [],
        "EMERGENCY_VEHICLE": [],
        "OTHER": [],
    }
    # create dictionary with keys 0-50
    histogram = {i: 0 for i in range(0, 80)}
    for input_folder_path_labels in input_folder_paths_all:
        input_files_labels = sorted(os.listdir(input_folder_path_labels))
        num_frames += len(input_files_labels)
        for label_file_name in input_files_labels:
            # R00 release
            # num_labeled_objects_one_frame = {
            #     "CAR": 0,
            #     "TRUCK": 0,
            #     "TRAILER": 0,
            #     "VAN": 0,
            #     "MOTORCYCLE": 0,
            #     "BUS": 0,
            #     "PEDESTRIAN": 0,
            #     "BICYCLE": 0,
            #     "SPECIAL_VEHICLE": 0,
            #     "OTHER": 0,
            # }
            # R01, R02, R03, R04 release
            num_labeled_objects_one_frame = {
                "CAR": 0,
                "TRUCK": 0,
                "TRAILER": 0,
                "VAN": 0,
                "MOTORCYCLE": 0,
                "BUS": 0,
                "PEDESTRIAN": 0,
                "BICYCLE": 0,
                "EMERGENCY_VEHICLE": 0,
                "OTHER": 0,
            }
            json_file = open(
                os.path.join(input_folder_path_labels, label_file_name),
            )
            json_data = json.load(json_file)
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                num_labeled_objects = len(frame_obj["objects"].keys())
                histogram[num_labeled_objects] += 1
                num_labeled_objects_all_frames_grouped.append(num_labeled_objects)
                for object_track_id, object_json in frame_obj["objects"].items():
                    object_class = object_json["object_data"]["type"]
                    num_labeled_objects_one_frame[object_class] += 1
            for object_class in num_labeled_objects_all_frames.keys():
                num_labeled_objects_all_frames[object_class].append(
                    num_labeled_objects_one_frame[object_class],
                )

    # group histogram by 5
    histogram_grouped = {}
    for key, value in histogram.items():
        histogram_grouped[key // 5] = histogram_grouped.get(key // 5, 0) + value

    avg_value = sum(num_labeled_objects_all_frames_grouped) / num_frames
    print(
        "num frames: %d, num total objects: %d, avg. num objects. per frame: %d"
        % (
            num_frames,
            sum(num_labeled_objects_all_frames_grouped),
            avg_value,
        )
    )
    print(num_labeled_objects_all_frames_grouped)
    num_labels_unique, inv_ndx = np.unique(num_labeled_objects_all_frames_grouped, return_inverse=True)
    print("num_labels_unique:", num_labels_unique)

    df = pandas.DataFrame(num_labeled_objects_all_frames_grouped, columns=["A"])
    df_grouped = df.groupby("A")
    print(df_grouped)
    df_bins_conted = df_grouped.size()
    print(df_bins_conted)

    ##########################################
    # 1. Histogram of the number of 3D box labels per frame
    ##########################################
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    # fig, ax = plt.subplots(figsize=(5, 4.5))
    # plt.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.12)

    # 16:10 = 8:5 = 4:2.5
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.subplots_adjust(left=0.17, right=0.99, top=0.95, bottom=0.2)
    # ax = plt.gca()
    num_bins = 10
    y_values, bins = np.histogram(num_labeled_objects_all_frames_grouped, bins=num_bins)
    # y_max = np.max(y_values)
    # create values from 0 to 1000 in 100 step size
    # TODO: do not hardcode range
    color_bar_labels = np.arange(0, 40, 10)
    bins = np.arange(0, 55, 5)
    y_values, bins = np.histogram(num_labeled_objects_all_frames_grouped, bins=bins)
    y_max = np.max(y_values)
    # round y_max to next 50
    y_max = int(math.ceil((y_max) / 50.0)) * 50
    step_size = int(y_max / 5)
    PlotUtils.plot_histogram(
        ax,
        num_labeled_objects_all_frames_grouped,
        num_bins=num_bins,
        range_list=[0, 50],
        x_label=r"\# 3D box labels",
        y_label=r"\# frames",
        bin_labels=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        y_max=y_max,
        step_size=step_size,
        color_bar_labels=None,
        font_size=font_size,
    )

    for y_value, bin in zip(y_values, bins):
        if bin < 10:
        # elif bin > 35:
            x_pos = bin
        else:
            x_pos = bin + 1
            # x_pos = bin
        ax.text(x_pos, y_value + y_max * 0.03, str(y_value), color="black", fontweight="bold", fontsize=font_size)
    # show average
    plt.axvline(x=avg_value, linewidth=1, color="r", linestyle="--", zorder=3)
    plt.text(avg_value - 5, y_max +200, str(round(avg_value)), color="r", fontsize=font_size)
    # show total number of labels
    plt.text(
        19,
        y_max+200,
        "Total 3D box labels: " + str(round(sum(num_labeled_objects_all_frames_grouped), 2)),
        color="black",
        fontsize=font_size,
    )

    # set y ticks from 0 to 350

    plt.yticks(np.arange(0, y_max+2*step_size, step_size))

    # change the fontsize of minor ticks label
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.tick_params(axis="both", which="minor", labelsize=font_size-2)

    plt.savefig(os.path.join(output_folder_path_statistic_plots, "histogram_objects_in_frame.pdf"))
    plt.close()
    plt.clf()

    ##########################################
    # 2. Box plot for avg number of objects per frame for each class (10 box plots)
    ##########################################
    # x-axis: 10 classes (color coded)
    # y-axis: box plots (min_number_of_objects_in_frame, avg_num_objects_in_frame, max_number_of_objects_in_frame). use standard deviation for start and end of box.
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.4)
    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.ylabel(r"\# 3D box labels per frame")
    plt.yticks(np.arange(0, 30, 2))

    data = []
    medians = []
    for object_class in num_labeled_objects_all_frames.keys():
        data.append(num_labeled_objects_all_frames[object_class])
        medians.append(np.median(num_labeled_objects_all_frames[object_class]))

    box_plot = plt.boxplot(
        data,
        patch_artist=True,  # fill with color
        vert=True,  # vertical box alignment
        medianprops=dict(color=(255 / 255, 0 / 255, 0 / 255, 1.0)),
        showfliers=False,
    )
    plt.setp(box_plot["boxes"], color="black")
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    for patch, color in zip(box_plot["boxes"], class_colors):
        patch.set_facecolor(color)
    xtickNames = plt.setp(ax, xticklabels=num_labeled_objects_all_frames.keys())
    plt.setp(xtickNames, rotation=45, fontsize=8)
    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(len(num_labeled_objects_all_frames.keys())) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ["bold", "semibold"]
    top = 23.5
    for tick, label in zip(range(10), ax.get_xticklabels()):
        k = tick % 2
        ax.text(
            pos[tick],
            top - (top * 0.05),
            upperLabels[tick],
            horizontalalignment="center",
            size="x-small",
            weight=weights[k],
            color="black",
        )
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "box_plot_objects_in_frame_train.pdf"))
