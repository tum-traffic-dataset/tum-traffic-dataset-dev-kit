import argparse
import glob
import os
import json
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
from internal.src.statistics.plot_utils import PlotUtils
from src.utils.vis_utils import VisualizationUtils

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
        help="Path to output folder",
        default="",
    )
    args = arg_parser.parse_args()
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots

    if not os.path.exists(output_folder_path_statistic_plots):
        os.makedirs(output_folder_path_statistic_plots)

    input_folder_paths_all = []
    if args.input_folder_path_labels:
        input_folder_paths_all.append(args.input_folder_path_labels)

    classes_list = ["CAR", "TRUCK", "TRAILER", "VAN", "MOTORCYCLE", "BUS", "PEDESTRIAN", "BICYCLE", "EMERGENCY_VEHICLE",
                    "OTHER"]

    for input_folder_path_labels in input_folder_paths_all:
        for label_file_name in sorted(os.listdir(input_folder_path_labels)):

            # iterate classes_list and create a dictionary with empty lists for each class
            classes = {}
            for class_name in classes_list:
                classes[class_name] = []

            class_colors = PlotUtils.get_class_colors(alpha=0.5)

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
            total_num_points = 0
            alpha = 0.5
            num_points_per_class = {}

            json_file = open(
                os.path.join(input_folder_path_labels, label_file_name),
            )
            json_data = json.load(json_file)
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                for object_track_id, object_json in frame_obj["objects"].items():
                    object_data = object_json["object_data"]
                    cuboid = object_data["cuboid"]["val"]
                    location = cuboid[:3]
                    distance = np.linalg.norm(location)
                    occlusion_attribute = VisualizationUtils.get_attribute_by_name(
                        object_data["cuboid"]["attributes"]["text"], "occlusion_level"
                    )

                    num_points_attribute = VisualizationUtils.get_attribute_by_name(
                        object_data["cuboid"]["attributes"]["num"], "num_points"
                    )
                    number_points = 0 if num_points_attribute["val"] == -1 else num_points_attribute["val"]
                    if object_data["type"] in classes.keys():
                        classes[object_data["type"]].append(number_points)
                    else:
                        classes[object_data["type"]] = [number_points]

            total_3d_box_labels = sum([len(num_points) for num_points in classes.values()])
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": "Computer Modern Roman",
                }
            )
            fig, ax = plt.subplots(figsize=(4, 2.5))
            plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.41)
            class_names = classes.keys()
            # rename EMERGENCY_VEHICLE to EMERGENCY_VEH
            class_names = [class_name if class_name != "EMERGENCY_VEHICLE" else "EMERGENCY" for class_name in class_names]

            occurrences = [len(class_num_points) for class_num_points in classes.values()]
            average_class_occurrences = np.mean(occurrences)
            print("Average number of labels per class: ", average_class_occurrences)
            plt.bar(
                class_names,
                occurrences,
                color=class_colors,
                edgecolor="black",
                zorder=3,
            )

            for i, num_labels in enumerate(occurrences):
                ax.text(i, num_labels + 2, str(num_labels), color="black", fontweight="bold",
                        fontsize=12)

            ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5, zorder=0)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            use_log_scale = False
            if use_log_scale:
                ax.set_yscale("log")
                ax.set_yticks([0.8, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])
                # Define a function to format the y-axis tick labels
                def log_tick_formatter(val, pos=None):
                    if val < 1:
                        return 0
                    else:
                        return r"$10^{{{}}}$".format(int(np.log10(val)))


                ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
            else:
                ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 35])
            plt.ylabel(r"\# 3D box labels")
            plt.text(0, 27,
                     "3D box labels: " + str(total_3d_box_labels),
                     color="black",
                     fontweight="bold",
                     fontsize=12)
            # save figure in 1920x1200 resolution
            plt.savefig(os.path.join(output_folder_path_statistic_plots, label_file_name.replace(".json", ".jpg")),
                        dpi=480)
            plt.close()
            plt.clf()
