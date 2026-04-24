import argparse
import glob
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from internal.src.statistics.plot_utils import PlotUtils
from src.utils.vis_utils import VisualizationUtils
from src.visualization.visualize_image_with_3d_boxes import set_track_history

if __name__ == "__main__":
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
        "--output_folder_path_statistic_plots",
        type=str,
        help="Path to output folder for statistic plots",
        default="output/statistic_plots",
    )
    arg_parser.add_argument(
        "--font_size",
        type=int,
        help="Font size for the plots",
        default=9,
    )
    # add sensor station id
    arg_parser.add_argument(
        "--sensor_station_id",
        type=str,
        help="Sensor station id. Examples: [s040, s050, s110]",
        default="s110",
    )
    args = arg_parser.parse_args()
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots
    font_size = args.font_size
    sensor_station_id = args.sensor_station_id

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

    classes_list = ["CAR", "TRUCK", "TRAILER", "VAN", "MOTORCYCLE", "BUS", "PEDESTRIAN", "BICYCLE", "EMERGENCY_VEHICLE",
                    "OTHER"]

    # iterate classes_list and create a dictionary with empty lists for each class
    classes = {}
    for class_name in classes_list:
        classes[class_name] = []

    class_colors = PlotUtils.get_class_colors(alpha=0.3)

    # automatically find out what object classes are present in the dataset
    classes_valid_set = set()
    valid_ids = set()
    for input_folder_path in input_folder_paths_all:
        label_file_paths = sorted(glob.glob(input_folder_path + "/*.json"))
        for label_file_path in label_file_paths:
            labels_json = json.load(open(label_file_path, "r"))
            for frame_idx, frame_obj in labels_json["openlabel"]["frames"].items():
                for uuid, box in frame_obj["objects"].items():
                    object_class = box["object_data"]["type"]
                    classes_valid_set.add(object_class)
                    valid_ids.add(classes_list.index(object_class))

    # remove not valid classes from classes
    classes_valid_list = list(classes_valid_set)
    class_coler_ids_to_delete = []
    for class_name in list(classes):
        if class_name not in classes_valid_list:
            del classes[class_name]
            # delete not valid color from class_colors
            class_coler_ids_to_delete.append(classes_list.index(class_name))

    class_colors = np.delete(class_colors, class_coler_ids_to_delete, axis=0)

    track_lengths_in_meters = {}
    avg_track_length_in_meters = 0.0
    total_track_length_in_meters = 0.0
    max_track_length_in_meters = 0.0

    track_lengths_in_frames = {}
    avg_track_length_in_frames = 0
    total_track_length_in_frames = 0
    max_track_length_in_frames = 0

    track_lengths_in_seconds = {}
    avg_track_length_in_seconds = 0.0
    total_track_length_in_seconds = 0.0
    max_track_length_in_seconds = 0.0

    for input_folder_path in tqdm(input_folder_paths_all):
        print("current folder path: {}".format(input_folder_path))
        if input_folder_path == "":
            continue
        label_file_paths = sorted(glob.glob(input_folder_path + "/*.json"))
        current_frame_idx = 0
        for label_file_path in tqdm(label_file_paths):
            labels_json = json.load(open(label_file_path, "r"))
            for frame_idx, frame_obj in labels_json["openlabel"]["frames"].items():
                for uuid, box in frame_obj["objects"].items():
                    track_history_attribute = VisualizationUtils.get_attribute_by_name(
                        box["object_data"]["cuboid"]["attributes"]["vec"], "track_history"
                    )
                    object_class = box["object_data"]["type"]
                    track_history = track_history_attribute["val"]
                    if uuid in track_lengths_in_meters.keys():
                        # update if length is longer
                        if len(track_history) > len(track_lengths_in_meters[str(uuid)][1]):
                            track_lengths_in_meters[str(uuid)] = [object_class, track_history]
                    else:
                        track_lengths_in_meters[str(uuid)] = [object_class, track_history]
                    if uuid in track_lengths_in_frames.keys():
                        # update if length is longer
                        if len(track_history) > len(track_lengths_in_meters[str(uuid)][1]):
                            track_lengths_in_frames[str(uuid)] = track_history

            current_frame_idx += 1

    min_track_length_each_class = []
    avg_track_length_each_class = []
    max_track_length_each_class = []
    for uuid, [object_class, track_history] in track_lengths_in_meters.items():
        if len(track_history) > 0:
            locations = np.reshape(track_history, (-1, 3))
            # calculate track length
            track_length_in_meter = 0.0
            for i in range(1, len(locations)):
                track_length_in_meter += np.linalg.norm(locations[i] - locations[i - 1])

            classes[object_class].append(track_length_in_meter)

            track_lengths_in_frames[uuid] = len(track_history) / 3
            if track_length_in_meter > max_track_length_in_meters:
                max_track_length_in_meters = track_length_in_meter
            if len(track_history) / 3 > max_track_length_in_frames:
                max_track_length_in_frames = len(track_history) / 3
            total_track_length_in_meters += track_length_in_meter

    # use classes dict and average all track lengths
    for object_class, track_lengths in classes.items():
        if len(track_lengths) > 0:
            min_track_length_each_class.append(np.min(track_lengths))
            avg_track_length_each_class.append(np.mean(track_lengths))
            max_track_length_each_class.append(np.max(track_lengths))
            print("class: {}, avg track length: {}".format(object_class, np.mean(track_lengths)))
            print("class: {}, max track length: {}".format(object_class, np.max(track_lengths)))
        else:
            print("class: {}, avg track length: {}".format(object_class, 0.0))
            min_track_length_each_class.append(0.0)
            avg_track_length_each_class.append(0.0)
            max_track_length_each_class.append(0.0)
    avg_track_length_in_meters = np.mean(avg_track_length_each_class)
    print("avg track length: {}".format(avg_track_length_in_meters))

    # print avg track length
    # print num of tracks
    print("num. of unique objects: ", len(track_lengths_in_meters))

    # original calculation
    # avg_track_length_in_meters = total_track_length_in_meters / len(track_lengths_in_meters.keys())

    # calculate average track length in meters
    # track_lengths_in_meters_list = []
    # for uuid, [object_class, track_history] in track_lengths_in_meters.items():
    #     if len(track_history) > 0:
    #         locations = np.reshape(track_history, (-1, 3))
    #         # calculate track length
    #         track_length_in_meter = 0.0
    #         for i in range(1, len(locations)):
    #             track_length_in_meter += np.linalg.norm(locations[i] - locations[i - 1])
    #         track_lengths_in_meters_list.append(track_length_in_meter)
    # avg_track_length_in_meters = np.mean(track_lengths_in_meters_list)

    print("avg track length (in meters): ", avg_track_length_in_meters)
    print("max track length (in meters): ", max_track_length_in_meters)
    print("total track length (in meters): ", total_track_length_in_meters)
    # calculate average track length in frames
    for uuid, track_length_in_frames in track_lengths_in_frames.items():
        if track_length_in_frames > 0:
            total_track_length_in_frames += track_length_in_frames
    avg_track_length_in_frames = total_track_length_in_frames / len(track_lengths_in_frames.keys())
    print("avg track length (in frames): ", round(avg_track_length_in_frames))
    print("max track length (in frames): ", int(max_track_length_in_frames))

    track_lengths_list = []
    for uuid, [object_class, track_history] in track_lengths_in_meters.items():
        if len(track_history) > 0:
            locations = np.reshape(track_history, (-1, 3))
            # calculate track length
            track_length_in_meter = 0.0
            for i in range(1, len(locations)):
                track_length_in_meter += np.linalg.norm(locations[i] - locations[i - 1])
            track_lengths_list.append(track_length_in_meter)
    # total track length:
    print("total track length (in m): ", sum(track_lengths_list))

    ##############################################
    # 1. Bar chart of the avg. track length for each class
    ##############################################
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    # fig, ax = plt.subplots(figsize=(5, 4.0))
    # plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.27)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.subplots_adjust(left=0.17, right=0.99, top=0.95, bottom=0.35)
    class_names = classes.keys()
    # change class name of EMERGENCY_VEHICLE to EMERGENCY_VEH
    class_names = [name.replace("EMERGENCY_VEHICLE", "EMERGENCY") for name in class_names]
    # remove class OTHER if test_sequence is used
    remove_other_class = False
    if remove_other_class:
        class_names.remove("OTHER")

        # add legend
    legend_elements = []
    legend_elements.append(Patch(facecolor="white", edgecolor="black", label="Max.", hatch=""))
    legend_elements.append(Patch(facecolor="white", edgecolor="black", label="Avg.", hatch="///"))
    # legend_elements.append(Patch(facecolor="white", edgecolor="black", label="Min.", hatch=""))
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=font_size - 2)

    # plot max track length
    plt.bar(
        class_names,
        max_track_length_each_class,
        color=class_colors,
        edgecolor="black",
        zorder=3,
        # alpha=0.5,
    )
    # plot bar chart for avg. track length
    plt.bar(
        class_names,
        avg_track_length_each_class,
        color=class_colors,
        edgecolor="black",
        zorder=3,
        # alpha=0.5,
        hatch="///"
    )
    # plot text labels for avg. track length
    for i, (max_track, avg_track) in enumerate(zip(max_track_length_each_class, avg_track_length_each_class)):
        # if max_track != avg_track:
        #     delta_y = 150
        # else:
        if max_track - avg_track < 20 and max_track != avg_track:
            delta_y = 25
        else:
            delta_y = 0
        ax.text(
            i - 0.25,
            avg_track + avg_track/7 - delta_y,
            str(round(avg_track)),
            color="black",
            fontweight="bold",
            fontsize=font_size,
        )
    # plot text labels for max. track length
    for i, max_track in enumerate(max_track_length_each_class):
        ax.text(
            i - 0.25,
            max_track + max_track/7,
            str(round(max_track)),
            color="black",
            fontweight="bold",
            fontsize=font_size
        )
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5, zorder=0)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(r"Track length [in m]", fontsize=font_size)
    use_log_scale = True
    if use_log_scale:
        ax.set_yscale("log")
        # ax.set_yticks([0.8, 5, 10, 20, 40, 80, 160, 320, 640,1280,2560])
        # set ticks from 0.8 to 10000
        #ax.set_yticks([0.8, 10, 100, 1000, 10000])
        # set y ticks based on max y value
        y_max_label = int(np.ceil(max(max_track_length_each_class) / 10000.0)) * 10000
        # calculate log scale ticks
        y_ticks = []
        y_ticks.append(0.8)
        for i in range(1, 10):
            # append until y_max is reached
            if 10 ** i < y_max_label:
                y_ticks.append(10 ** i)
            else:
                break
        ax.set_yticks(y_ticks)



        # Define a function to format the y-axis tick labels
        # def log_tick_formatter(val, pos=None):
        #     if val < 1:
        #         return 0
        #     else:
        #         return str(int(val))

        def log_tick_formatter(val, pos=None):
            if val < 1:
                return 0
            # else:
            value = int(np.log10(int(val)))
            return str(r"$10^{%01d}$" % value)


        # Set the y-axis tick labels to show the actual values
        ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    else:
        # set y ticks
        y_ticks = np.arange(0, 220, 20)
        plt.yticks(y_ticks)
    plt.axhline(y=avg_track_length_in_meters, linewidth=1, color="r", linestyle="--", zorder=3)
    plt.text(
        4 -0.25,
        avg_track_length_in_meters + 150,
        str(round(avg_track_length_in_meters)),
        color="r",
        fontweight="bold",
        fontsize=font_size,
    )

    plt.text(-0.5, avg_track_length_in_meters + 3000,
             "Total track length: " + str(round(sum(track_lengths_list) / 1000, 2)) + " km",
             color="black",
             fontweight="bold", fontsize=font_size)

    # change the fontsize of minor ticks label
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.tick_params(axis="both", which="minor", labelsize=font_size - 2)
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "bar_chart_avg_track_lengths_all_drives.pdf"))
    plt.close()
    plt.clf()

    ##############################################
    # 2. Histogram of the track lengths
    ##############################################
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    # fig, ax = plt.subplots(figsize=(5, 4.5))
    # plt.subplots_adjust(left=0.2, right=0.99, top=0.97, bottom=0.12)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.21)
    use_log_scale = True
    if sensor_station_id == "s110":
        bin_labels = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
        bins = np.arange(0, 170, 10).tolist()
        num_bins = 17
        range_list = [0, 170]
    elif sensor_station_id == "s050" or sensor_station_id == "s040":
        # create bins from 0 to 700 in 50 steps
        bin_labels = np.arange(0, 800, 100).tolist()
        bins = np.arange(0, 750, 50).tolist()
        num_bins = 14
        range_list = [0, 700]
    else:
        raise ValueError("Sensor station id not supported")

    y_values = PlotUtils.plot_histogram(
        ax,
        track_lengths_list,
        num_bins=num_bins,
        range_list=range_list,
        x_label="Track length [m]",
        y_label="Num. of tracks",
        bin_labels=bin_labels,
        use_log_scale=use_log_scale,
        step_size=None,
        font_size=font_size,
        color_bar_labels=None,
    )
    if use_log_scale:
        ax.set_yscale("log")
        ax.set_yticks([0.8, 1, 5, 10, 50, 100, 200])
        # set y ticks based on max y value
        y_max_label = int(np.ceil(np.max(y_values) / 1000.0)) * 1000
        # calculate log scale ticks
        y_ticks = []
        y_ticks.append(0.8)
        for i in range(1, 10):
            # append until y_max is reached
            if 10 ** i < y_max_label:
                y_ticks.append(10 ** i)
            else:
                break
        # ax.set_yticks(y_ticks)


        # Define a function to format the y-axis tick labels
        def log_tick_formatter(val, idx):
            print("val", val)
            if val < 1:
                return 0
            else:
                return str(int(val))


        # Set the y-axis tick labels to show the actual values
        ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    else:
        # ceil to the nearest 10
        y_max_label = int(np.ceil(np.max(y_values) / 10.0)) * 10 + 10
        ax.set_yticks(np.arange(0, y_max_label, 5))

    # show y values
    # for i in range(len(y_values)):
    #     ax.text(i+2, y_values[i]+y_values[i]/10, str(y_values[i]), color="black", fontweight="bold",
    #             fontsize=font_size)
    # show y values
    idx = 0
    for y_value, bin_current in zip(y_values, bins):
        x_pos = None
        if idx == 0 or idx == len(y_values) - 1:
            x_pos = bin_current
        elif idx + 1 < len(y_values) and y_value < y_values[idx + 1] and y_value > y_values[idx - 1]:
            x_pos = bin_current - 4
        elif idx + 1 < len(y_values) and y_value >= y_values[idx + 1] and y_value < y_values[idx - 1]:
            x_pos = bin_current + 4
        else:
            x_pos = bin_current + 3

        ax.text(x_pos, y_value + y_value / 5, str(int(y_value)), color="black", fontweight="bold", fontsize=font_size - 3)
        idx += 1


    # show average
    plt.axvline(x=avg_track_length_in_meters, linewidth=1, color="r", linestyle="--", zorder=3)
    plt.text(
        avg_track_length_in_meters - 30, np.max(y_values), str(round(avg_track_length_in_meters, 2)) + " m", color="r", fontsize=font_size
    )

    # show total track length in meters and kilo meters
    plt.text(
        60,
        120,
        "Total track length: " + str(round(sum(track_lengths_list) / 1000, 2)) + " km",
        color="black",
        fontsize=font_size,
    )
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.tick_params(axis="both", which="minor", labelsize=font_size - 2)
    plt.savefig(os.path.join(output_folder_path_statistic_plots, "histogram_track_lengths.pdf"))
    plt.close()
