import os
import json
import matplotlib.pyplot as plt

input_files_path = [
    "/home/user/updated_label_track_id_01_05_23/r01_full_registered_point_cloud_labels_train_01_05_23/s110_lidar_ouster_south/",
    "/home/user/updated_label_track_id_01_05_23/r01_full_registered_point_cloud_labels_val_01_05_23/s110_lidar_ouster_south/",
    "/home/user/updated_label_track_id_01_05_23/r01_full_registered_point_cloud_labels_test_01_05_23/s110_lidar_ouster_south/sampled/",
    "/home/user/updated_label_track_id_01_05_23/r01_full_registered_point_cloud_labels_test_01_05_23/s110_lidar_ouster_south/sequence/",
]

weather_types = dict()
day_times = dict()

for input_files_labels in input_files_path:
    for label_file_name in sorted(os.listdir(input_files_labels)):
        json_file = open(
            os.path.join(input_files_labels, label_file_name),
        )
        json_data = json.load(json_file)
        for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
            frame_properties = frame_obj["frame_properties"]
            weather_type = frame_properties["weather_type"]
            day_time = frame_properties["time_of_day"]
            if weather_type in weather_types.keys():
                weather_types[str(weather_type)] += 1
            else:
                weather_types[str(weather_type)] = 1
            if day_time in day_times.keys():
                day_times[str(day_time)] += 1
            else:
                day_times[str(day_time)] = 1

print("Weather types:", weather_types)

fig = plt.figure()
plt.bar(weather_types.keys(), weather_types.values())
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.3)
plt.savefig("weather_types_statistics.png")
plt.clf()

print("Day times:", day_times)

fig = plt.figure()
plt.bar(day_times.keys(), day_times.values())
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.3)
plt.savefig("day_of_time_statistics.png")
