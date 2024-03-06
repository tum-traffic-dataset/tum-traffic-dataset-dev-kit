import os
import json
from pathlib import Path
import numpy as np

import uuid
from internal.src.tracking.tracker import BoxTracker
from src.utils.detection import save_to_openlabel, Detection
from src.utils.vis_utils import VisualizationUtils
from scipy.spatial.transform import Rotation as R
import argparse

def generate_uuids(num_uuids):
    uuids = []
    for i in range(num_uuids):
        uuids.append(str(uuid.uuid4()))
    return uuids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path_boxes",
        type=str,
        help="Input directory path",
        default="",
    )
    parser.add_argument(
        "--output_folder_path_boxes_tracked",
        type=str,
        help="Output directory path",
        default="",
    )
    parser.add_argument(
        "--max_age",
        type=int,
        help="Max age of detections (time to live) in frames",
        default=0,
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_folder_path_boxes_tracked):
        os.makedirs(args.output_folder_path_boxes_tracked)

    mapping_id_to_uuid = generate_uuids(10000)

    # TODO:
    # create SDRT tracker
    tracker = BoxTracker(max_prediction_age=int(args.max_age))
    # create PolyMOT tracker
    #tracker = PolyMOTTracker(max_prediction_age=int(args.max_age))
    for file_name in sorted(os.listdir(args.input_folder_path_boxes)):
        label_data = json.load(open(os.path.join(args.input_folder_path_boxes, file_name)))
        detections = []
        frame_id_str = None
        frame_properties = None
        if "coordinate_systems" in label_data["openlabel"]:
            coordinate_systems = label_data["openlabel"]["coordinate_systems"]
        else:
            coordinate_systems = {}
        if "streams" in label_data["openlabel"]:
            streams = label_data["openlabel"]["streams"]
        else:
            streams = None
        id = 0
        for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
            frame_id_str = str(frame_id)
            if "frame_properties" in frame_obj:
                frame_properties = frame_obj["frame_properties"]
            else:
                frame_properties = None
            for object_id, label in frame_obj["objects"].items():
                cuboid = np.array(label["object_data"]["cuboid"]["val"])
                if "attributes" in label["object_data"]["cuboid"]:
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "sensor_id"
                    )
                    if attribute is not None:
                        sensor_id = attribute["val"]
                    else:
                        sensor_id = ""
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "body_color"
                    )
                    if attribute is not None:
                        color = attribute["val"]
                    else:
                        color = ""
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["num"], "num_points"
                    )
                    if attribute is not None:
                        num_points = int(float(attribute["val"]))
                    else:
                        num_points = -1
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["num"], "score"
                    )
                    if attribute is not None:
                        score = round(float(attribute["val"]), 2)
                    else:
                        score = -1
                    attribute = VisualizationUtils.get_attribute_by_name(
                        label["object_data"]["cuboid"]["attributes"]["text"], "occlusion_level"
                    )
                    if attribute is not None:
                        occlusion_level = attribute["val"]
                    else:
                        occlusion_level = "NOT_OCCLUDED"
                else:
                    sensor_id = ""
                    color = ""
                    num_points = -1
                    score = -1
                detections.append(
                    Detection(
                        id=id,
                        uuid=object_id,
                        category=label["object_data"]["type"],
                        location=np.array([[cuboid[0]], [cuboid[1]], [cuboid[2]]]),
                        dimensions=(cuboid[7], cuboid[8], cuboid[9]),
                        yaw=R.from_quat(np.array([cuboid[3], cuboid[4], cuboid[5], cuboid[6]])).as_euler(
                            "xyz", degrees=False
                        )[2],
                        score=score,
                        num_lidar_points=num_points,
                        color=color,
                        occlusion_level=occlusion_level,
                        sensor_id=sensor_id,
                    )
                )
                id += 1

        # sort tracking
        detections = tracker.update(detections)

        # update uuids
        for detection in detections:
            detection.uuid = mapping_id_to_uuid[detection.id]

        save_to_openlabel(
            detections,
            file_name,
            Path(args.output_folder_path_boxes_tracked),
            coordinate_systems,
            frame_properties,
            frame_id_str,
            streams,
        )
