import os
import json
import argparse

""" This script converts YOLO annotation files to OpenLABEL annotation files
YOLO Object detection format:        <class-index> <x_center> <y_center> <width> <height
YOLO Instance segmentation format:   <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>

Example execution:
----------
python YOLO_to_OPENLABEL.py 
    / --input_yolo_folder_path  yolo_labels 
    / --output_folder_path  yolo_to_openlabel_labels 
    / --sensor_id "s110_camera_basler_south2_8mm" 

"""

# current matching of category_id -> category:   0 -> "car", 1 -> "truck", 2 -> "van", ...
# change if other order was defined for the yolo label files
CATEGORY = ["car", "truck", "trailer", "van", "motorcycle", "bus", "pedestrian", "bicycle", "emergency vehicle",
            "other"]


def one_yolo_file_to_openlabel(
        input_YOLO_annotation_file: str,  # .txt file
        output_folder_path: str,
        frame_id: str,
        img_width: int,
        img_height: int,
        sensor_id: str,
        coordinate_systems=None,
        frame_properties=None,
        streams=None,
):
    """
    Convert one annotation file in yolo format to OpenLABEL format
    """

    output_json_data = {"openlabel": {"metadata": {"schema_version": "1.0.0"}, "coordinate_systems": {}}}
    if coordinate_systems:
        output_json_data["openlabel"]["coordinate_systems"] = coordinate_systems

    objects_map = {}
    frame_map = {str(frame_id): {}}
    object_id = 0

    # open the .txt yolo file, each line is one object
    with open(input_YOLO_annotation_file, "r") as f:
        for line in f:
            # print("line: ", line)

            ## line.split()[0] := category index
            category = CATEGORY[int(line.split()[0])]
            # print("category: ", category)

            ## line.split()[1:] := normalized polygon : x1 y1 x2 y2 ... xn yn
            polygon_or_bbox = []
            for p in line.split()[1:]:
                polygon_or_bbox.append(float(p))
            polygon_or_bbox = [int(p * img_width) if i % 2 == 0 else int(p * img_height) for i, p in
                               enumerate(polygon_or_bbox)]

            # if polygon_or_bbox has more than 4 points, it is a polygon, otherwise it is a bbox
            if len(polygon_or_bbox) > 4:
                polygon = polygon_or_bbox
                ## create bounding box for this polygon
                x_min = min(polygon[::2])
                x_max = max(polygon[::2])
                y_min = min(polygon[1::2])
                y_max = max(polygon[1::2])
                width = x_max - x_min
                height = y_max - y_min
                x_center = x_min + width / 2.0
                y_center = y_min + height / 2.0
                bbox = [int(x_center), int(y_center), int(width), int(height)]

                objects_map[object_id] = {
                    "object_data": {
                        "name": category.upper() + "_" + str(object_id),
                        "type": category.upper(),
                        "poly2d": [
                            {
                                "name": "mask",  # full mask or visible mask depending on the yolo file
                                "val": polygon,
                                "attributes": {
                                    "text": [
                                        {
                                            "name": "sensor_id",
                                            "val": sensor_id
                                        }
                                    ]
                                }
                            }
                        ],
                        "bbox": [
                            {
                                "name": "full_bbox",  # full bbox or visible bbox depending on the yolo file
                                "val": bbox,
                                "attributes": {
                                    "text": [
                                        {
                                            "name": "sensor_id",
                                            "val": sensor_id
                                        }
                                    ]
                                }
                            }
                        ]

                    }
                }
            else:
                bbox = polygon_or_bbox
                objects_map[object_id] = {
                    "object_data": {
                        "name": category.upper() + "_" + str(object_id),
                        "type": category.upper(),
                        "bbox": [
                            {
                                "name": "full_bbox",  # full bbox or visible bbox depending on the yolo file
                                "val": bbox,
                                "attributes": {
                                    "text": [
                                        {
                                            "name": "sensor_id",
                                            "val": sensor_id
                                        }
                                    ]
                                }
                            }
                        ]

                    }
                }

            # print("bbox: ", bbox)
            # print("polygon: ", polygon)
            object_id += 1

    frame_map[frame_id]["objects"] = objects_map
    if frame_properties:
        frame_map[frame_id]["frame_properties"] = frame_properties
    if streams:
        output_json_data["openlabel"]["streams"] = streams
    output_json_data["openlabel"]["frames"] = frame_map

    with open(output_folder_path + "/" + input_YOLO_annotation_file.split("/")[-1].split(".")[0] + ".json", "w",
              encoding="utf-8") as f:
        json.dump(output_json_data, f, indent=4)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_yolo_folder_path", type=str,
                            help="Path to folder containing yolo-formated annotations", default="")
    arg_parser.add_argument("--output_folder_path", type=str, help="Path to OPENLabel output folder", default="")
    arg_parser.add_argument("--sensor_id", type=str, help="Sensor ID", default="")
    arg_parser.add_argument("--img_width", type=str, help="image width", default=1920)
    arg_parser.add_argument("--img_height", type=str, help="image height", default=1200)
    args = arg_parser.parse_args()
    sensor_id = args.sensor_id
    input_yolo_folder_path = args.input_yolo_folder_path
    output_folder_path = args.output_folder_path
    img_width = args.img_width
    img_height = args.img_height

    # create output folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    frame_id = 0
    # convert each yolo file to one openlabel file and save it in the output folder
    for file in os.listdir(input_yolo_folder_path):
        if file.endswith(".txt"):
            yolo_file_path = os.path.join(input_yolo_folder_path, file)
            print("Converting {}".format(yolo_file_path))
            one_yolo_file_to_openlabel(yolo_file_path, output_folder_path, str(frame_id), img_width, img_height,
                                       sensor_id, frame_id)

            frame_id += 1

