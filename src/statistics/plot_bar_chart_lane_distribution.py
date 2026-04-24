import argparse
import glob
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def get_lane_id_by_y_position(pos_y):
    lane_id = None
    # if 2.1 <= pos_y < 5.9:
    if 1.5 <= pos_y < 5.9:
        lane_id = -1
    elif 5.9 <= pos_y < 9.4:
        lane_id = -2
    elif 9.4 <= pos_y < 13.15:
        lane_id = -3
    elif 13.15 <= pos_y < 16.9:
        lane_id = -4
    elif 16.9 <= pos_y < 20.65:
        lane_id = -5
    # elif 20.65 <= pos_y < 23.9:
    elif 20.65 <= pos_y:
        lane_id = -6
    elif -5.9 <= pos_y < -2.1:
        lane_id = 1
    elif -9.4 <= pos_y < -5.9:
        lane_id = 2
    elif -13.15 <= pos_y < -9.4:
        lane_id = 3
    elif -16.9 <= pos_y < -13.15:
        lane_id = 4
    elif -20.65 <= pos_y < -16.9:
        lane_id = 5
    # elif -23.9 <= pos_y < -20.65:
    elif pos_y < -20.65:
        lane_id = 6
    return lane_id


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_labels",
        type=str,
        help="Path to labels",
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
    args = arg_parser.parse_args()
    input_folder_path_labels = args.input_folder_path_labels
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots
    font_size = args.font_size

    if not os.path.exists(output_folder_path_statistic_plots):
        os.makedirs(output_folder_path_statistic_plots)

    histogram_lane_distribution = {}
    label_file_paths = sorted(glob.glob(input_folder_path_labels + "/*.json"))
    for label_file_path in label_file_paths:
        labels_json = json.load(open(label_file_path, "r"))
        for frame_idx, frame_obj in labels_json["openlabel"]["frames"].items():
            for uuid, box in frame_obj["objects"].items():
                cuboid = box["object_data"]["cuboid"]["val"]
                location_3d = cuboid[0:3]
                y_position = location_3d[1]
                # get lane ID by y position
                lane_id = get_lane_id_by_y_position(y_position)
                if lane_id is not None:
                    if lane_id not in histogram_lane_distribution:
                        histogram_lane_distribution[lane_id] = 0
                    histogram_lane_distribution[lane_id] += 1
                else:
                    print("No lane ID found for y position: ", y_position)
    # plot histogram
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )

    # sort keys of histogram_lane_distribution
    histogram_lane_distribution[0] = 0  # green lane
    histogram_lane_distribution = dict(sorted(histogram_lane_distribution.items()))
    fig, ax = plt.subplots(figsize=(4, 2.5))
    #plt.subplots_adjust(left=0.15, right=0.99, top=0.95, bottom=0.16)
    plt.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.21)
    bin_labels = list(histogram_lane_distribution.keys())
    all_values = list(histogram_lane_distribution.values())
    num_bins = len(bin_labels)
    range_list = (min(bin_labels), max(bin_labels))
    x_label = "Lane ID"
    y_label = "Number of objects"
    bin_labels = bin_labels
    use_log_scale = False
    y_max = None
    step_size = None
    color_bar_labels = None
    font_size = font_size
    ax.set_xticks(bin_labels)
    if use_log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
        # set plot y ticks based on the max y value
        y_max = max(all_values)
        y_max = int(np.ceil(y_max / 1000)) * 1000
        ax.set_yticks(np.arange(0, y_max + 1, 200))

    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    #y_max = np.max(n) if y_max is None else y_max
    y_max = max(all_values) if y_max is None else y_max
    # round to next 2
    y_max = int(np.ceil(y_max / 2.0)) * 2
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)

    for i, (bin_label, num_labels) in enumerate(zip(bin_labels, all_values)):
        # shift the text label to the right if the y value of i+i is smaller than the y value of i
        shift = 0
        # shift the text label to the right if the y value of bin_label+i is smaller than the y value of bin_label
        if num_labels != 0 and i < len(all_values) - 1:
            if num_labels < all_values[i+1]:
                shift = -0.5
            # do not shift the text label if the y value of bin_label+1 is smaller and the y values of bin_label-1 is also smaller
            elif all_values[i + 1] < num_labels and all_values[i - 1] < num_labels:
                shift = 0
            # shift the text label to the left if the y value of bin_label+1 is larger than the y value of bin_label
            elif all_values[i+1] < num_labels:
                shift = 0.0




        ax.text(bin_label-0.3+shift, num_labels + 20, str(num_labels), color="black", fontweight="bold",
                fontsize=font_size)
    # create a color map (red to blue) and color each bar
    cm = plt.cm.get_cmap("RdYlBu_r")
    # Get the colormap colors
    my_cmap = cm(np.arange(cm.N))
    # Set alpha
    my_cmap[:, -1] = np.linspace(0.5, 0.5, cm.N)
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    # color each bar
    for i, (bin_label, num_labels) in enumerate(zip(bin_labels, all_values)):
        color = cm((num_labels - min(all_values)) / (max(all_values) - min(all_values)))
        # make color more transparent
        color = (color[0], color[1], color[2], 0.5)
        plt.bar(bin_label, num_labels, color=color, edgecolor="black", zorder=3)
    plt.text(-3, 1850,
             "Number of objects: " + str(sum(all_values)),
             color="black",
             fontweight="bold",
             fontsize=font_size)
    # add text label for driving direction north and south
    plt.text(5, 1650,
             "North",
             color="black",
             fontweight="bold",
             fontsize=font_size)
    plt.text(-5, 1650,
                "South",
                color="black",
                fontweight="bold",
                fontsize=font_size)
    # add color bar on the right side of the plot
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=min(all_values), vmax=max(all_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.ax.tick_params(labelsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.savefig(os.path.join(output_folder_path_statistic_plots, "histogram_lane_distribution.pdf"))
    plt.close()
