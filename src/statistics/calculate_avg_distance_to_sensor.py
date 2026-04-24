import numpy as np
import open3d as o3d
import json
import argparse
import os
import pandas

pandas.set_option('display.max_rows', None)


# Example:
#  python calculate_avg_distance_to_sensor.py --labels_folder_path1 /mnt/hdd_data1/28_datasets/00_a9_dataset/00_R0/01_R0_S1/04_labels/ --labels_folder_path2 /mnt/hdd_data1/28_datasets/00_a9_dataset/00_R0/02_R0_S2/04_labels/


def calculate_avg_distance_to_sensor(labels_folder_paths, lens_type):
    avg_points_cars_all_frames = []
    num_cars_all_frames = 0
    for labels_folder_path in labels_folder_paths:
        label_files = sorted(os.listdir(labels_folder_path))
        file_idx = 0
        for label_file_name in label_files:
            print("processing frame: ", str(file_idx))

            if lens_type in label_file_name:
                json_file = open(os.path.join(labels_folder_path, label_file_name), )
                json_data = json.load(json_file)
                num_cars = 0
                for label in json_data["labels"]:
                    # print("processing label: ", str(label_idx))
                    x = float(label["box3d"]["position"]["x"])
                    y = float(label["box3d"]["position"]["y"])
                    z = float(label["box3d"]["position"]["z"])

                if num_cars > 0:
                    avg_points_cars = round(points_inside_car / num_cars)
                    avg_points_cars_all_frames.append(avg_points_cars)
                    num_cars_all_frames = num_cars_all_frames + num_cars
                    points_inside_car_all_frames = points_inside_car_all_frames + points_inside_car
                    points_inside_special_vehicle_all_frames = points_inside_special_vehicle_all_frames + points_inside_special_vehicle

                file_idx = file_idx + 1

    if len(avg_points_cars_all_frames) > 0:
        print("num frames total: %d" % (len(avg_points_cars_all_frames)))

    num_points_unique_car, _ = np.unique(avg_points_cars_all_frames, return_inverse=True)
    print(num_points_unique_car)
    df_car = pandas.DataFrame(avg_points_cars_all_frames, columns=['A'])
    df_grouped_car = df_car.groupby('A')
    print("car", str(df_grouped_car))
    df_bins_conted_car = df_grouped_car.size()
    print(df_bins_conted_car.to_string())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Argument Parser')
    argparser.add_argument('--labels_folder_path1', default="labels1",
                           help='Folder path of labels. Default: labels1')
    argparser.add_argument('--labels_folder_path2', default="labels2",
                           help='Folder path of labels. Default: labels2')
    argparser.add_argument('--lens_type', default="16mm",
                           help='lens type. Can be one of [16mm, 50mm]. Default: 16mm')
    args = argparser.parse_args()
    labels_folder_path1 = args.labels_folder_path1
    labels_folder_path2 = args.labels_folder_path2
    lens_type = args.lens_type

    labels_folder_paths = [labels_folder_path1, labels_folder_path2]
    calculate_avg_distance_to_sensor(labels_folder_paths,lens_type)
