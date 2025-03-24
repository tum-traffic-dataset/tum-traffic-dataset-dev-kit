import os
import json
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path_boxes",
        type=str,
        help="Input directory path",
        default="a9_dataset/r01_full_split/test_full/detections_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered",
    )
    parser.add_argument(
        "--output_folder_path_boxes_combined",
        type=str,
        help="Output directory path",
        default="a9_dataset/r01_full_split/test_full/detections_point_clouds/",
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_folder_path_boxes_combined):
        os.makedirs(args.output_folder_path_boxes_combined)
    frames = {}
    for frame_idx, file_name in enumerate(sorted(os.listdir(args.input_folder_path_boxes))):
        label_data = json.load(open(os.path.join(args.input_folder_path_boxes, file_name)))
        for _, frame_obj in label_data["openlabel"]["frames"].items():
            frames[str(frame_idx)] = frame_obj
    output_json_data = {"openlabel": {"metadata": {"schema_version": "1.0.0"}, "coordinate_systems": {}}}
    output_json_data["openlabel"]["frames"] = frames
    with open(os.path.join(args.output_folder_path_boxes_combined, "submission.json"), 'w',
              encoding='utf-8') as json_writer:
        json_string = json.dumps(output_json_data, ensure_ascii=True, indent=4)
        json_writer.write(json_string)
