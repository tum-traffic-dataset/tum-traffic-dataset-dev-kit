import os
import numpy as np

input_folder = '/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_ouster_south/'

for filename in os.listdir(input_folder):
    with open(os.path.join(input_folder, filename), 'r') as file_reader:
        lines = file_reader.readlines()
        lines_keep = []
        ring_array = np.zeros((64, 1), dtype=int)
        for idx, line in enumerate(lines):
            if idx < 11:
                continue
            parts = line.split(" ")
            if (int(float(parts[0])) == 0 and int(float(parts[1])) == 0 and int(float(parts[2])) == 0):
                ring_array[int(parts[6])] = ring_array[int(parts[6])] + 1
                continue

        print(ring_array)
        break
