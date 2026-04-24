import argparse
import glob
import os
import json
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from internal.src.statistics.plot_utils import PlotUtils
from src.utils.vis_utils import VisualizationUtils


def draw_heatmap(image, points_2d, sigma, output_file_path_statistics):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.125))
    heatmap, xedges, yedges = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=400, range=[[0, 1920], [0, 1200]])
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    heatmap_img = heatmap.T
    ax.imshow(heatmap_img, extent=extent, cmap=cm.jet)
    ax.imshow(image, alpha=0.5)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1200, 0)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.savefig(output_file_path_statistics, dpi=384)
    plt.close(fig)


if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_images",
        type=str,
        help="Path to images",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels",
        type=str,
        help="Path to labels",
        default="",
    )
    arg_parser.add_argument(
        "--output_folder_path_statistics",
        type=str,
        help="Path to output folder",
        default="",
    )
    args = arg_parser.parse_args()
    output_folder_path_statistics = args.output_folder_path_statistics
    input_folder_path_images = args.input_folder_path_images
    input_folder_path_labels = args.input_folder_path_labels

    # extract sensor ID from input_folder_path_images
    camera_sensor_id = input_folder_path_images.split("/")[-1]

    if not os.path.exists(output_folder_path_statistics):
        os.makedirs(output_folder_path_statistics)

    classes_list = ["CAR", "TRUCK", "TRAILER", "VAN", "MOTORCYCLE", "BUS", "PEDESTRIAN", "BICYCLE", "EMERGENCY_VEHICLE",
                    "OTHER"]
    # iterate classes_list and create a dictionary with empty lists for each class
    classes = {}
    for class_name in classes_list:
        classes[class_name] = []

    class_colors = PlotUtils.get_class_colors(alpha=0.5)

    # automatically find out what object classes are present in the dataset
    classes_valid_set = set()
    valid_ids = set()
    label_file_paths = sorted(glob.glob(input_folder_path_labels + "/*.json"))
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

    if input_folder_path_images is not None:
        input_file_paths_images = sorted(glob.glob(input_folder_path_images + "/*"))
    else:
        input_file_paths_images = [None] * len(input_folder_path_labels)

    utils = VisualizationUtils()
    sigma = 8
    idx = 0
    reference_image = None
    points_2d_all = []
    for label_file_path, input_file_path_image in zip(label_file_paths, input_file_paths_images):
        label_file_name = os.path.basename(label_file_path)
        json_file = open(
            os.path.join(input_folder_path_labels, label_file_name),
        )
        json_data = json.load(json_file)
        # load image
        image = None
        if input_file_path_image is not None:
            image = cv2.imread(os.path.join(input_file_path_image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points_2d = []
        if idx == 0:
            reference_image = image
        for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
            for object_track_id, object_json in frame_obj["objects"].items():
                object_data = object_json["object_data"]
                cuboid = object_data["cuboid"]["val"]
                location = cuboid[:3]
                # make location a 3,1 array
                location = np.expand_dims(location, axis=1)
                # project 3d points to 2d
                point_2d = utils.project_3d_box_to_2d(location.T, camera_sensor_id,
                                                      "s110_lidar_ouster_south",
                                                      boxes_coordinate_system_origin="s110_lidar_ouster_south")
                # continue if empty
                if point_2d is None:
                    continue

                # make integer
                point_2d = point_2d.astype(int)
                # check if point is in image
                img_shape = image.shape
                img_width = img_shape[1]
                img_height = img_shape[0]
                if point_2d[0] < 0 or point_2d[0] > img_width or point_2d[1] < 0 or point_2d[1] > img_height:
                    continue
                points_2d.append(point_2d)
                points_2d_all.append(point_2d)

        points_2d_all_arr = np.array(points_2d_all)
        points_2d_all_arr = points_2d_all_arr.reshape(-1, 2)
        output_file_path_statistics = os.path.join(output_folder_path_statistics,
                                                   label_file_name.replace(".json", ".jpg"))
        draw_heatmap(image, points_2d_all_arr, sigma, output_file_path_statistics)
        idx += 1
    points_2d_all = np.array(points_2d_all)
    points_2d_all = points_2d_all.reshape(-1, 2)
    output_file_path_statistics = os.path.join(output_folder_path_statistics, "all.jpg")
    draw_heatmap(reference_image, points_2d_all, sigma, output_file_path_statistics)
