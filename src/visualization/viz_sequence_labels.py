import json
import cv2
import argparse
import sys
import os

# Visualize labels with a 3D cuboid
# Example:
# python viz_sequence_labels.py --image_sequence_folder_path images --label_folder_path labels --output_folder_path visualization

def draw_line(img, start_point, end_point, color):
    img = cv2.line(img, start_point, end_point, color, 2)


class Utils:

    def __init__(self):
        pass

    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))




if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='VizLabel Argument Parser')
    argparser.add_argument('-i', '--image_sequence_folder_path', default="images",
                           help='Image sequence folder path. Default: images')
    argparser.add_argument('-l', '--label_folder_path', default="labels", help='Label folder path. Default: labels')
    argparser.add_argument('-o', '--output_folder_path',
                           help='Output folder path to save visualization results to disk.')
    args = argparser.parse_args()

    image_sequence_folder_path = args.image_sequence_folder_path
    label_folder_path = args.label_folder_path
    output_folder_path = args.output_folder_path

    utils = Utils()

    image_file_paths = sorted(os.listdir(image_sequence_folder_path))
    label_file_paths = sorted(os.listdir(label_folder_path))

    if len(image_file_paths) != len(label_file_paths):
        print("Error: Make sure the number of image files matches the number of label files.")
        sys.exit()

    file_idx = 0
    for image_file_path, label_file_path in zip(image_file_paths, label_file_paths):
        print("Processing image file: " + image_file_path)
        file_idx = file_idx + 1
        img = cv2.imread(os.path.join(image_sequence_folder_path, image_file_path), cv2.IMREAD_UNCHANGED)

        data = open(os.path.join(label_folder_path, label_file_path), )
        labels = json.load(data)

        for box_3d_label in labels["labels"]:
            color = None
            if box_3d_label["category"] == "CAR":
                color = "#00CCF6"
            elif box_3d_label["category"] == "TRUCK":
                color = "#56FFB6"
            elif box_3d_label["category"] == "TRAILER":
                color = "#5AFF7E"
            elif box_3d_label["category"] == "VAN":
                color = "#EBCF36"
            elif box_3d_label["category"] == "MOTORCYCLE":
                color = "#B9A454"
            elif box_3d_label["category"] == "BUS":
                color = "#D98A86"
            elif box_3d_label["category"] == "PEDESTRIAN":
                color = "#E976F9"
            elif box_3d_label["category"] == "BICYCLE":
                color = "#B18CFF"
            elif box_3d_label["category"] == "SPECIAL_VEHICLE":
                color = "#C7C7C7"
            color = utils.hex_to_rgb(color)
            color = (color[2], color[1], color[0])
            utils.draw_3d_box_by_keypoints(img, box_3d_label, color)

        if output_folder_path:
            if not os.path.isdir(output_folder_path):
                os.mkdir(output_folder_path)
            cv2.imwrite(
                output_folder_path + "/" + image_file_path.split(".")[0] + "_with_labels." + image_file_path.split(".")[
                    1], img)
        else:
            cv2.imshow(str(box_3d_label["id"]), img)
            cv2.waitKey()
