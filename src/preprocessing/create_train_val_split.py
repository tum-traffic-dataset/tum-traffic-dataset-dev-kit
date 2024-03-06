import argparse
import glob
import os

from tqdm import tqdm


# Module description:
# This script is used to create a train/val split for the dataset using existing file indices stored inside the data_split folder.
# The final train/val split is stored in a new folder:  <INPUT_FOLDER_PATH_DATASET>+ "_split"
#
# Usage: python src/preprocessing/create_train_val_split.py --input_folder_path_dataset <INPUT_FOLDER_PATH_DATASET> --input_folder_path_data_split_root <INPUT_FOLDER_PATH_DATA_SPLIT_ROOT>
# Example: python src/preprocessing/create_train_val_split.py --input_folder_path_dataset /home/user/datasets/r02_sequences --input_folder_path_data_split_root /home/user/a9-dataset-dev-kit/data_split

def copy_to_output(
        file_name,
        dataset_root_folder_path,
        subsets,
        sensor_modality,
        sensors,
        output_folder_path,
):
    for subset in subsets:
        for sensor in sensors:
            input_folder_path = os.path.join(dataset_root_folder_path, subset, sensor_modality, sensor)
            file_paths_sub_set = sorted(glob.glob(input_folder_path + "/*"))
            for file_path_sub_set in file_paths_sub_set:
                file_name_in_sub_set = os.path.basename(file_path_sub_set)
                if file_name == file_name_in_sub_set:
                    input_file_path = os.path.join(input_folder_path, file_name)
                    output_file_path = os.path.join(output_folder_path, file_name)
                    os.system(f"cp {input_file_path} {output_file_path}")
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path_dataset",
        type=str,
        help="Folder path to dataset root.",
        default="",
    )
    parser.add_argument(
        "--input_folder_path_data_split_root",
        type=str,
        help="Folder path to dev-kit data_split root.",
        default="",
    )

    args = parser.parse_args()
    dataset_root_folder_path = args.input_folder_path_dataset
    input_folder_path_data_split_root = args.input_folder_path_data_split_root
    output_folder_path = dataset_root_folder_path + "_split"
    dataset_output_sub_sets = ["train", "val"]
    subsets = ["a9_dataset_r02_s01", "a9_dataset_r02_s02", "a9_dataset_r02_s03", "a9_dataset_r02_s04"]
    sensor_modalities = ["point_clouds", "images", "labels_point_clouds"]
    camera_sensors = ["s110_camera_basler_south2_8mm", "s110_camera_basler_south1_8mm"]
    lidar_sensors = ["s110_lidar_ouster_south", "s110_lidar_ouster_north"]
    for dataset_output_sub_set in dataset_output_sub_sets:
        input_folder_path_dataset_sub_set = os.path.join(dataset_root_folder_path, dataset_output_sub_set)
        # crate output folder
        output_folder_path_dataset_sub_set = output_folder_path + "/" + dataset_output_sub_set
        os.makedirs(output_folder_path_dataset_sub_set, exist_ok=True)
        for sensor_modality in sensor_modalities:
            input_folder_path_sensor_modality = os.path.join(input_folder_path_dataset_sub_set, sensor_modality)
            # create output folder
            output_folder_path_dataset_sub_set_sensor_modality = (
                    output_folder_path_dataset_sub_set + "/" + sensor_modality
            )
            os.makedirs(output_folder_path_dataset_sub_set_sensor_modality, exist_ok=True)
            sensors = None
            if sensor_modality == "point_clouds":
                sensors = lidar_sensors
            elif sensor_modality == "images":
                sensors = camera_sensors
            elif sensor_modality == "labels_point_clouds":
                sensors = lidar_sensors
            else:
                raise ValueError("Invalid sensor modality.")
            for sensor in sensors:
                input_folder_path_lidar_sensor = os.path.join(input_folder_path_sensor_modality, sensor)
                # create output folder
                output_folder_path_dataset_sub_set_sensor_modality_lidar_sensor = (
                        output_folder_path_dataset_sub_set_sensor_modality + "/" + sensor
                )
                os.makedirs(output_folder_path_dataset_sub_set_sensor_modality_lidar_sensor, exist_ok=True)
                with open(
                        f"{input_folder_path_data_split_root}/{dataset_output_sub_set}/{sensor_modality}/{sensor}/file_names.txt",
                        "r",
                ) as file:
                    file_names = file.readlines()
                for file_name in tqdm(file_names):
                    file_name = file_name.strip()
                    copy_to_output(
                        file_name,
                        dataset_root_folder_path,
                        subsets,
                        sensor_modality,
                        sensors,
                        output_folder_path_dataset_sub_set_sensor_modality_lidar_sensor,
                    )
