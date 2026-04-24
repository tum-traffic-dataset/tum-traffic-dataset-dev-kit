import argparse
import math
import os
import numpy as np


def calculate_synchronization_statistics(input_folder_path1, input_folder_path2):
    sync_error_list = []
    for filename1, filename2 in zip(sorted(os.listdir(input_folder_path1)), sorted(os.listdir(input_folder_path2))):
        nano_seconds1 = int(filename1.split("_")[0]) * 1000000000 + int(filename1.split("_")[1])
        nano_seconds2 = int(filename2.split("_")[0]) * 1000000000 + int(filename2.split("_")[1])
        sync_error = abs(nano_seconds1 - nano_seconds2) / 1000000
        sync_error_list.append(sync_error)
        # print("synchronization error (in ms) between " + filename1 + " and " + filename2 + ": ",sync_error)
    min = np.min(np.array(sync_error_list))
    avg = np.sum(np.array(sync_error_list)) / len(sync_error_list)
    max = np.max(np.array(sync_error_list))
    # get indices of min and max
    min_index = np.argmin(np.array(sync_error_list))
    max_index = np.argmax(np.array(sync_error_list))
    return min, avg, max, min_index, max_index


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder_path_infrastructure_camera_south1", type=str, help="input folder path south1", default="")
    parser.add_argument("--input_folder_path_infrastructure_camera_south2", type=str, help="input folder path south2", default="")
    parser.add_argument("--input_folder_path_infrastructure_camera_north", type=str, help="input folder path north", default="")
    parser.add_argument("--input_folder_path_infrastructure_camera_east", type=str, help="input folder path east", default="")
    parser.add_argument("--input_folder_path_infrastructure_lidar_south", type=str, help="input folder path south", default="")
    parser.add_argument("--input_folder_path_infrastructure_lidar_north", type=str, help="input folder path north", default="")
    parser.add_argument("--input_folder_path_vehicle_lidar_robosense", type=str, help="input folder path to vehicle lidar robosense", default="")
    parser.add_argument("--input_folder_path_vehicle_camera_basler", type=str,
                        help="input folder path to vehicle camera basler", default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # parse parameters
    args = parse_arguments()

    input_folder_paths_all = []
    if args.input_folder_path_infrastructure_camera_south1 != "":
        input_folder_paths_all.append(args.input_folder_path_infrastructure_camera_south1)
    if args.input_folder_path_infrastructure_camera_south2 != "":
        input_folder_paths_all.append(args.input_folder_path_infrastructure_camera_south2)
    if args.input_folder_path_infrastructure_camera_north != "":
        input_folder_paths_all.append(args.input_folder_path_infrastructure_camera_north)
    if args.input_folder_path_infrastructure_camera_east != "":
        input_folder_paths_all.append(args.input_folder_path_infrastructure_camera_east)
    if args.input_folder_path_infrastructure_lidar_south != "":
        input_folder_paths_all.append(args.input_folder_path_infrastructure_lidar_south)
    if args.input_folder_path_infrastructure_lidar_north != "":
        input_folder_paths_all.append(args.input_folder_path_infrastructure_lidar_north)
    if args.input_folder_path_vehicle_lidar_robosense != "":
        input_folder_paths_all.append(args.input_folder_path_vehicle_lidar_robosense)
    if args.input_folder_path_vehicle_camera_basler != "":
        input_folder_paths_all.append(args.input_folder_path_vehicle_camera_basler)


    # calculate total synchronization errors between all input folder paths
    sync_errors = []
    total_min_sync_error = -1
    total_avg_sync_error = -1
    avg_sync_errors = []
    total_max_sync_error = -1

    for i in range(len(input_folder_paths_all)):
        for j in range(i + 1, len(input_folder_paths_all)):
            min, avg, max, _, _ = calculate_synchronization_statistics(input_folder_paths_all[i], input_folder_paths_all[j])
            sync_errors.append({"min": min, "avg": avg, "max": max})
            last_folder_name1 = input_folder_paths_all[i].split("/")[-1]
            last_folder_name2 = input_folder_paths_all[j].split("/")[-1]
            print("sync error between ", last_folder_name1, " and ", last_folder_name2, " (in ms): min: ", min, " avg: ", avg, " max: ", max)
            if total_min_sync_error == -1:
                total_min_sync_error = min
            else:
                total_min_sync_error = min if min < total_min_sync_error else total_min_sync_error

            if total_max_sync_error == -1:
                total_max_sync_error = max
            else:
                total_max_sync_error = max if max > total_max_sync_error else total_max_sync_error

            avg_sync_errors.append(avg)


    # print synchronization errors
    print("total min sync error (in ms): ", total_min_sync_error)
    print("total avg sync error (in ms): ", np.sum(np.array(avg_sync_errors)) / len(avg_sync_errors))
    print("total max sync error (in ms): ", total_max_sync_error)








    # 1. synchronization error (camera south1 to camera south2)

    # min, avg, max, _, _ = calculate_synchronization_statistics(input_folder_path_south1, input_folder_path_south2)
    # camera_sync_errors = {"min": min, "avg": avg, "max": max}
    # print("sync error between south1 and south2 (in ms): min: ", min, " avg: ", avg, " max: ", max)

    # 2. synchronization error (lidar south to lidar north)
    # min, avg, max, _, _ = calculate_synchronization_statistics(input_folder_path_south, input_folder_path_north)
    # lidar_sync_errors = {"min": min, "avg": avg, "max": max}
    # print("sync error between south and north LiDAR (in ms): min: ", min, " avg: ", avg, " max: ", max)

    # 3. synchronization error (lidar to camera)
    # min, avg, max, _, _ = calculate_synchronization_statistics(input_folder_path_south1, input_folder_path_north)
    # camera_to_lidar_sync_errors = {
    #     "min": np.min([camera_sync_errors["min"], lidar_sync_errors["min"]]),
    #     "avg": (camera_sync_errors["avg"] + lidar_sync_errors["avg"]) / 2.0,
    #     "max": np.max([camera_sync_errors["max"], lidar_sync_errors["max"]]),
    # }
    # print("sync error between south1 and north (in ms): min: ", min, " avg: ", avg, " max: ", max)
    #
    # # 4. synchronization error (total)
    # min, avg, max, min_idx, max_idx = calculate_synchronization_statistics(
    #     input_folder_path_south1, input_folder_path_south
    # )
    # camera_south1_lidar_south_sync_errors = {"min": min, "avg": avg, "max": max}
    # print("sync error between south1 camera and south LiDAR (in ms): min: ", min, " avg: ", avg, " max: ", max)
    # # print indices
    # print("min index: ", min_idx, " max index: ", max_idx)
    #
    # min, avg, max, _, _ = calculate_synchronization_statistics(input_folder_path_south1, input_folder_path_north)
    # camera_south1_lidar_north_sync_errors = {"min": min, "avg": avg, "max": max}
    # print("sync error between south1 camera and north LiDAR (in ms): min: ", min, " avg: ", avg, " max: ", max)
    #
    # min, avg, max, _, _ = calculate_synchronization_statistics(input_folder_path_south2, input_folder_path_south)
    # camera_south2_lidar_south_sync_errors = {"min": min, "avg": avg, "max": max}
    # print("sync error between south2 camera and south LiDAR (in ms): min: ", min, " avg: ", avg, " max: ", max)
    #
    # min, avg, max, _, _ = calculate_synchronization_statistics(input_folder_path_south2, input_folder_path_north)
    # camera_south2_lidar_north_sync_errors = {"min": min, "avg": avg, "max": max}
    # print("sync error between south2 camera and north LiDAR (in ms): min: ", min, " avg: ", avg, " max: ", max)
    #
    # total_sync_errors = {
    #     "min": np.min(
    #         [
    #             camera_sync_errors["min"],
    #             lidar_sync_errors["min"],
    #             camera_south1_lidar_south_sync_errors["min"],
    #             camera_south1_lidar_north_sync_errors["min"],
    #             camera_south2_lidar_south_sync_errors["min"],
    #             camera_south2_lidar_north_sync_errors["min"],
    #         ]
    #     ),
    #     "avg": np.average(
    #         [
    #             camera_sync_errors["avg"],
    #             lidar_sync_errors["avg"],
    #             camera_south1_lidar_south_sync_errors["avg"],
    #             camera_south1_lidar_north_sync_errors["avg"],
    #             camera_south2_lidar_south_sync_errors["avg"],
    #             camera_south2_lidar_north_sync_errors["avg"],
    #         ]
    #     ),
    #     "max": np.max(
    #         [
    #             camera_sync_errors["max"],
    #             lidar_sync_errors["max"],
    #             camera_south1_lidar_south_sync_errors["max"],
    #             camera_south1_lidar_north_sync_errors["max"],
    #             camera_south2_lidar_south_sync_errors["max"],
    #             camera_south2_lidar_north_sync_errors["max"],
    #         ]
    #     ),
    # }
    # print("total sync error between all 4 sensors (in ms): min: ", min, " avg: ", avg, " max: ", max)
