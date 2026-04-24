import os
import json

input_files_labels = "/mnt/hdd_data1/28_datasets/00_a9_dataset/00_R0/02_R0_S2/04_labels/"
idx = 0
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

min_height = IMAGE_HEIGHT
for label_file_name in sorted(os.listdir(input_files_labels)):
    json_file = open(os.path.join(input_files_labels, label_file_name), )
    json_data = json.load(json_file)
    for label in json_data["labels"]:
        xmin, ymin, xmax, ymax = label["box2d"]
        if (ymax - ymin) < min_height:
            min_height = (ymax - ymin)

print("min height:")
print(min_height)
