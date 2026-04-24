import numpy as np
import open3d as o3d
import json
import argparse
import os
import pandas

pandas.set_option('display.max_rows', None)


# Example:
#  python count_points_within_box_statistics.py --point_cloud_folder_path1 /mnt/hdd_data1/28_datasets/00_a9_dataset/00_R0/03_R0_S3/03_point_clouds/ --point_cloud_folder_path2 /mnt/hdd_data1/28_datasets/00_a9_dataset/00_R0/04_R0_S4/03_point_clouds/ --labels_folder_path1 /mnt/hdd_data1/28_datasets/00_a9_dataset/00_R0/03_R0_S3/04_labels/ --labels_folder_path2 /mnt/hdd_data1/28_datasets/00_a9_dataset/00_R0/04_R0_S4/04_labels/


def box_center_to_corner(cx, cy, cz, l, w, h, rotation):
    translation = [cx, cy, cz]
    # Create a bounding box outline
    bounding_box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [0 / 2, 0 / 2, 0 / 2, 0 / 2, h, h, h, h]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))
    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
    return corner_box.transpose()


def point_inside_parallelogram(x, y, poly):
    inside = False
    xb = poly[0][0] - poly[1][0]
    yb = poly[0][1] - poly[1][1]
    xc = poly[2][0] - poly[1][0]
    yc = poly[2][1] - poly[1][1]
    xp = x - poly[1][0]
    yp = y - poly[1][1]
    d = xb * yc - yb * xc
    if d != 0:
        oned = 1.0 / d
        bb = (xp * yc - xc * yp) * oned
        cc = (xb * yp - xp * yb) * oned
        inside = (bb >= 0) & (cc >= 0) & (bb <= 1) & (cc <= 1)
    return inside


def count_points_within_box(corner_box, points):
    poly = []
    point_2d = [corner_box[0, 0], corner_box[0, 1]]
    poly.append(point_2d)
    point_2d = [corner_box[1, 0], corner_box[1, 1]]
    poly.append(point_2d)
    point_2d = [corner_box[2, 0], corner_box[2, 1]]
    poly.append(point_2d)
    num_points = 0
    for point in points:
        # TODO: check if point inside 3D box
        inside = point_inside_parallelogram(point[0], point[1], poly)
        if inside and point[2] >= corner_box[0, 2] and point[2] <= corner_box[4, 2]:
            num_points = num_points + 1
    return num_points


def calculate_avg_points_within_objects(point_cloud_folder_paths, labels_folder_paths):
    points_inside_object_list = []
    # avg_points_cars_all_frames = []
    # avg_points_trailers_all_frames = []
    # avg_points_trucks_all_frames = []
    # avg_points_vans_all_frames = []
    # avg_points_motorcycles_all_frames = []
    # avg_points_busses_all_frames = []
    # avg_points_pedestrians_all_frames = []
    # avg_points_bicycles_all_frames = []
    # avg_points_special_vehicles_all_frames = []
    # avg_points_objects_all_frames = []

    # num_cars_all_frames = 0
    # num_trailers_all_frames = 0
    # num_trucks_all_frames = 0
    # num_vans_all_frames = 0
    # num_motorcycles_all_frames = 0
    # num_busses_all_frames = 0
    # num_pedestrians_all_frames = 0
    # num_bicycles_all_frames = 0
    # num_special_vehicles_all_frames = 0
    # num_objects_all_frames = 0

    # points_inside_car_all_frames = 0
    # points_inside_trailer_all_frames = 0
    # points_inside_truck_all_frames = 0
    # points_inside_van_all_frames = 0
    # points_inside_motorcycle_all_frames = 0
    # points_inside_bus_all_frames = 0
    # points_inside_pedestrian_all_frames = 0
    # points_inside_bicycle_all_frames = 0
    # points_inside_special_vehicle_all_frames = 0
    # points_inside_objects_all_frames = 0

    for point_cloud_folder_path, labels_folder_path in zip(point_cloud_folder_paths, labels_folder_paths):
        point_cloud_files = sorted(os.listdir(point_cloud_folder_path))
        label_files = sorted(os.listdir(labels_folder_path))
        file_idx = 0
        for point_cloud_file, label_file in zip(point_cloud_files, label_files):
            print("processing frame: ", str(file_idx))
            pcd = o3d.io.read_point_cloud(os.path.join(point_cloud_folder_path, point_cloud_file))
            points = pcd.points

            json_file = open(os.path.join(labels_folder_path, label_file), )
            json_data = json.load(json_file)

            # num_cars = 0
            # num_trailers = 0
            # num_trucks = 0
            # num_vans = 0
            # num_motorcycles = 0
            # num_busses = 0
            # num_pedestrians = 0
            # num_bicycles = 0
            # num_special_vehicles = 0
            num_objects = 0

            # points_inside_car = 0
            # points_inside_trailer = 0
            # points_inside_truck = 0
            # points_inside_van = 0
            # points_inside_motorcycle = 0
            # points_inside_bus = 0
            # points_inside_pedestrian = 0
            # points_inside_bicycle = 0
            # points_inside_special_vehicle = 0
            points_inside_object = 0

            # label_idx = 0
            for label in json_data["labels"]:
                # print("processing label: ", str(label_idx))
                l = float(label["box3d"]["dimension"]["length"])
                w = float(label["box3d"]["dimension"]["width"])
                h = float(label["box3d"]["dimension"]["height"])
                rotation = float(label["box3d"]["orientation"]["rotationYaw"])
                position_3d = [float(label["box3d"]["location"]["x"]), float(label["box3d"]["location"]["y"]),
                               float(label["box3d"]["location"]["z"])]

                corner_box = box_center_to_corner(position_3d[0], position_3d[1], position_3d[2], l, w, h, rotation)
                num_points_inside_box = count_points_within_box(corner_box, points)

                print("num points inside %s: %d" % (label["category"], num_points_inside_box))
                points_inside_object_list.append(num_points_inside_box)
                # label_idx = label_idx + 1
                # if label["category"] == "CAR":
                #     points_inside_car = points_inside_car + num_points_inside_box
                #     num_cars = num_cars + 1
                # elif label["category"] == "TRAILER" or label["category"] == "TRUCK":
                #     points_inside_trailer = points_inside_trailer + num_points_inside_box
                #     num_trailers = num_trailers + 1
                #     points_inside_truck = points_inside_truck + num_points_inside_box
                #     num_trucks = num_trucks + 1
                # elif label["category"] == "VAN":
                #     points_inside_van = points_inside_van + num_points_inside_box
                #     num_vans = num_vans + 1
                # elif label["category"] == "MOTORCYCLE":
                #     points_inside_motorcycle = points_inside_motorcycle + num_points_inside_box
                #     num_motorcycles = num_motorcycles + 1
                # elif label["category"] == "BUS":
                #     points_inside_bus = points_inside_bus + num_points_inside_box
                #     num_busses = num_busses + 1
                # elif label["category"] == "PEDESTRIAN":
                #     points_inside_pedestrian = points_inside_pedestrian + num_points_inside_box
                #     num_pedestrians = num_pedestrians + 1
                # elif label["category"] == "BICYCLE":
                #     points_inside_bicycle = points_inside_bicycle + num_points_inside_box
                #     num_pedestrians = num_pedestrians + 1
                # elif label["category"] == "SPECIAL_VEHICLE":
                #     points_inside_special_vehicle = points_inside_special_vehicle + num_points_inside_box
                #     num_special_vehicles = num_special_vehicles + 1
            # calculate average
            # if num_cars > 0:
            #     avg_points_cars = round(points_inside_car / num_cars)
            #     avg_points_cars_all_frames.append(avg_points_cars)
            #     num_cars_all_frames = num_cars_all_frames + num_cars
            #     points_inside_car_all_frames = points_inside_car_all_frames + points_inside_car
            # if num_trailers > 0:
            #     avg_points_trailers = round(points_inside_trailer / num_trailers)
            #     avg_points_trailers_all_frames.append(avg_points_trailers)
            #     num_trailers_all_frames = num_trailers_all_frames + num_trailers
            #     points_inside_trailer_all_frames = points_inside_trailer_all_frames + points_inside_trailer
            # if num_trucks > 0:
            #     avg_points_trucks = round(points_inside_truck / num_trucks)
            #     avg_points_trucks_all_frames.append(avg_points_trucks)
            #     num_trucks_all_frames = num_trucks_all_frames + num_trucks
            #     points_inside_truck_all_frames = points_inside_truck_all_frames + points_inside_truck
            # if num_vans > 0:
            #     avg_points_vans = round(points_inside_van / num_vans)
            #     avg_points_vans_all_frames.append(avg_points_vans)
            #     num_vans_all_frames = num_vans_all_frames + num_vans
            #     points_inside_van_all_frames = points_inside_van_all_frames + points_inside_van
            # if num_motorcycles > 0:
            #     avg_points_motorcycles = round(points_inside_motorcycle / num_motorcycles)
            #     avg_points_motorcycles_all_frames.append(avg_points_motorcycles)
            #     num_motorcycles_all_frames = num_motorcycles_all_frames + num_motorcycles
            #     points_inside_motorcycle_all_frames = points_inside_motorcycle_all_frames + points_inside_motorcycle
            # if num_busses > 0:
            #     avg_points_busses = round(points_inside_bus / num_busses)
            #     avg_points_busses_all_frames.append(avg_points_busses)
            #     num_busses_all_frames = num_busses_all_frames + num_busses
            #     points_inside_bus_all_frames = points_inside_bus_all_frames + points_inside_bus
            # if num_pedestrians > 0:
            #     avg_points_pedestrians = round(points_inside_pedestrian / num_pedestrians)
            #     avg_points_pedestrians_all_frames.append(avg_points_pedestrians)
            #     num_pedestrians_all_frames = num_pedestrians_all_frames + num_pedestrians
            #     points_inside_pedestrian_all_frames = points_inside_pedestrian_all_frames + points_inside_pedestrian
            # if num_bicycles > 0:
            #     avg_points_bicycles = round(points_inside_bicycle / num_bicycles)
            #     avg_points_bicycles_all_frames.append(avg_points_bicycles)
            #     num_bicycles_all_frames = num_bicycles_all_frames + num_bicycles
            #     points_inside_bicycle_all_frames = points_inside_bicycle_all_frames + points_inside_bicycle
            # if num_special_vehicles > 0:
            #     avg_points_special_vehicles = round(points_inside_special_vehicle / num_special_vehicles)
            #     avg_points_special_vehicles_all_frames.append(avg_points_special_vehicles)
            #     num_special_vehicles_all_frames = num_special_vehicles_all_frames + num_special_vehicles
            #     points_inside_special_vehicle_all_frames = points_inside_special_vehicle_all_frames + points_inside_special_vehicle
            # if num_objects > 0:
            # avg_points_objects = round(points_inside_object / num_objects)
            # avg_points_objects_all_frames.append(avg_points_objects)
            # num_objects_all_frames = num_objects_all_frames + num_objects
            # points_inside_objects_all_frames = points_inside_objects_all_frames + points_inside_object

            file_idx = file_idx + 1
        # print("finished with subset. Avg. Points for all frames of subset:")
        # print("car: ", str(avg_points_cars_all_frames))
        # print("trailer: ", str(avg_points_trailers_all_frames))
        # print("truck: ", str(avg_points_trucks_all_frames))
        # print("van: ", str(avg_points_vans_all_frames))
        # print("motorcycle: ", str(avg_points_motorcycles_all_frames))
        # print("bus: ", str(avg_points_busses_all_frames))
        # print("pedestrian: ", str(avg_points_pedestrians_all_frames))
        # print("bicycle: ", str(avg_points_bicycles_all_frames))
        # print("special_vehicle: ", str(avg_points_special_vehicles_all_frames))
        # print("objects: ", str(avg_points_objects_all_frames))

    # if len(avg_points_cars_all_frames) > 0:
    #     print("num frames total: %d, num total points inside cars: %d, avg. num points of car class: %d" % (
    #         len(avg_points_cars_all_frames), points_inside_car_all_frames,
    #         points_inside_car_all_frames / len(avg_points_cars_all_frames)))
    # if len(avg_points_trailers_all_frames) > 0:
    #     print("num frames total: %d, num total points inside trailers: %d, avg. num points of trailer class: %d" % (
    #         len(avg_points_trailers_all_frames), points_inside_trailer_all_frames,
    #         points_inside_trailer_all_frames / len(avg_points_trailers_all_frames)))
    # if len(avg_points_trucks_all_frames) > 0:
    #     print("num frames total: %d, num total points inside trucks: %d, avg. num points of truck class: %d" % (
    #         len(avg_points_trucks_all_frames), points_inside_truck_all_frames,
    #         points_inside_truck_all_frames / len(avg_points_trucks_all_frames)))
    # if len(avg_points_vans_all_frames) > 0:
    #     print("num frames total: %d, num total points inside vans: %d, avg. num points of van class: %d" % (
    #         len(avg_points_vans_all_frames), points_inside_van_all_frames,
    #         points_inside_van_all_frames / len(avg_points_vans_all_frames)))
    # if len(avg_points_motorcycles_all_frames) > 0:
    #     print(
    #         "num frames total: %d, num total points inside motorcycles: %d, avg. num points of motorcycle class: %d" % (
    #             len(avg_points_motorcycles_all_frames), points_inside_motorcycle_all_frames,
    #             points_inside_motorcycle_all_frames / len(avg_points_motorcycles_all_frames)))
    # if len(avg_points_busses_all_frames) > 0:
    #     print("num frames total: %d, num total points inside busses: %d, avg. num points of bus class: %d" % (
    #         len(avg_points_busses_all_frames), points_inside_bus_all_frames,
    #         points_inside_bus_all_frames / len(avg_points_busses_all_frames)))
    # if len(avg_points_pedestrians_all_frames) > 0:
    #     print(
    #         "num frames total: %d, num total points inside pedestrians: %d, avg. num points of pedestrian class: %d" % (
    #             len(avg_points_pedestrians_all_frames), points_inside_pedestrian_all_frames,
    #             points_inside_pedestrian_all_frames / len(avg_points_pedestrians_all_frames)))
    # if len(avg_points_bicycles_all_frames) > 0:
    #     print("num frames total: %d, num total points inside bicycles: %d, avg. num points of bicycle class: %d" % (
    #         len(avg_points_bicycles_all_frames), points_inside_bicycle_all_frames,
    #         points_inside_bicycle_all_frames / len(avg_points_bicycles_all_frames)))
    # if len(avg_points_special_vehicles_all_frames) > 0:
    #     print(
    #         "num frames total: %d, num total points inside special_vehicles: %d, avg. num points of special_vehicle class: %d" % (
    #             len(avg_points_special_vehicles_all_frames), points_inside_special_vehicle_all_frames,
    #             points_inside_special_vehicle_all_frames / len(avg_points_special_vehicles_all_frames)))
    # if len(avg_points_objects_all_frames) > 0:
    #     print(
    #         "num frames total: %d, num total points inside objects: %d, avg. num points of objects: %d" % (
    #             len(avg_points_objects_all_frames), points_inside_objects_all_frames,
    #             points_inside_objects_all_frames / len(avg_points_objects_all_frames)))

    # print(avg_points_cars_all_frames)
    # print(avg_points_trailers_all_frames)
    # print(avg_points_trucks_all_frames)
    # print(avg_points_vans_all_frames)
    # print(avg_points_motorcycles_all_frames)
    # print(avg_points_busses_all_frames)
    # print(avg_points_pedestrians_all_frames)
    # print(avg_points_bicycles_all_frames)
    # print(avg_points_special_vehicles_all_frames)
    # print(avg_points_objects_all_frames)

    # num_points_unique_car, _ = np.unique(avg_points_cars_all_frames, return_inverse=True)
    # num_points_unique_trailer, _ = np.unique(avg_points_trailers_all_frames, return_inverse=True)
    # num_points_unique_truck, _ = np.unique(avg_points_trucks_all_frames, return_inverse=True)
    # num_points_unique_van, _ = np.unique(avg_points_vans_all_frames, return_inverse=True)
    # num_points_unique_motorcycle, _ = np.unique(avg_points_motorcycles_all_frames, return_inverse=True)
    # num_points_unique_bus, _ = np.unique(avg_points_busses_all_frames, return_inverse=True)
    # num_points_unique_pedestrian, _ = np.unique(avg_points_pedestrians_all_frames, return_inverse=True)
    # num_points_unique_bicycle, _ = np.unique(avg_points_bicycles_all_frames, return_inverse=True)
    # num_points_unique_special_vehicle, _ = np.unique(avg_points_special_vehicles_all_frames, return_inverse=True)
    num_points_unique_objects, _ = np.unique(points_inside_object_list, return_inverse=True)

    # print(num_points_unique_car)
    # print(num_points_unique_trailer)
    # print(num_points_unique_truck)
    # print(num_points_unique_van)
    # print(num_points_unique_motorcycle)
    # print(num_points_unique_bus)
    # print(num_points_unique_pedestrian)
    # print(num_points_unique_bicycle)
    # print(num_points_unique_special_vehicle)
    print(num_points_unique_objects)

    # df_car = pandas.DataFrame(avg_points_cars_all_frames, columns=['A'])
    # df_trailer = pandas.DataFrame(avg_points_trailers_all_frames, columns=['A'])
    # df_truck = pandas.DataFrame(avg_points_trucks_all_frames, columns=['A'])
    # df_van = pandas.DataFrame(avg_points_vans_all_frames, columns=['A'])
    # df_motorcycle = pandas.DataFrame(avg_points_motorcycles_all_frames, columns=['A'])
    # df_bus = pandas.DataFrame(avg_points_busses_all_frames, columns=['A'])
    # df_pedestrian = pandas.DataFrame(avg_points_pedestrians_all_frames, columns=['A'])
    # df_bicycle = pandas.DataFrame(avg_points_bicycles_all_frames, columns=['A'])
    # df_special_vehicle = pandas.DataFrame(avg_points_special_vehicles_all_frames, columns=['A'])
    df_objects = pandas.DataFrame(points_inside_object_list, columns=['A'])

    # df_grouped_car = df_car.groupby('A')
    # df_grouped_trailer = df_trailer.groupby('A')
    # df_grouped_truck = df_truck.groupby('A')
    # df_grouped_van = df_van.groupby('A')
    # df_grouped_motorcycle = df_motorcycle.groupby('A')
    # df_grouped_bus = df_bus.groupby('A')
    # df_grouped_pedestrian = df_pedestrian.groupby('A')
    # df_grouped_bicycle = df_bicycle.groupby('A')
    # df_grouped_special_vehicle = df_special_vehicle.groupby('A')
    df_grouped_objects = df_objects.groupby('A')

    # print("car", str(df_grouped_car))
    # print("trailer", str(df_grouped_trailer))
    # print("truck", str(df_grouped_truck))
    # print("van", str(df_grouped_van))
    # print("motorcycle", str(df_grouped_motorcycle))
    # print("bus", str(df_grouped_bus))
    # print("pedestrian", str(df_grouped_pedestrian))
    # print("bicycle", str(df_grouped_bicycle))
    # print("special_vehicle", str(df_grouped_special_vehicle))
    print("objects", str(df_grouped_objects))

    # df_bins_conted_car = df_grouped_car.size()
    # df_bins_conted_trailer = df_grouped_trailer.size()
    # df_bins_conted_truck = df_grouped_truck.size()
    # df_bins_conted_van = df_grouped_van.size()
    # df_bins_conted_motorcycle = df_grouped_motorcycle.size()
    # df_bins_conted_bus = df_grouped_bus.size()
    # df_bins_conted_pedestrian = df_grouped_pedestrian.size()
    # df_bins_conted_bicycle = df_grouped_bicycle.size()
    # df_bins_conted_special_vehicle = df_grouped_special_vehicle.size()
    df_bins_conted_objects = df_grouped_objects.size()

    # print(df_bins_conted_car.to_string())
    # print(df_bins_conted_trailer.to_string())
    # print(df_bins_conted_truck.to_string())
    # print(df_bins_conted_van.to_string())
    # print(df_bins_conted_motorcycle.to_string())
    # print(df_bins_conted_bus.to_string())
    # print(df_bins_conted_pedestrian.to_string())
    # print(df_bins_conted_bicycle.to_string())
    # print(df_bins_conted_special_vehicle.to_string())
    print(df_bins_conted_objects.to_string())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Argument Parser')
    argparser.add_argument('--point_cloud_folder_path1', default="points1",
                           help='Point Cloud folder path. Default: points1')
    argparser.add_argument('--labels_folder_path1', default="labels1",
                           help='Folder path of labels. Default: labels1')
    argparser.add_argument('--point_cloud_folder_path2', default="points2",
                           help='Point Cloud folder path. Default: points2')
    argparser.add_argument('--labels_folder_path2', default="labels2",
                           help='Folder path of labels. Default: labels2')
    args = argparser.parse_args()
    point_cloud_folder_path1 = args.point_cloud_folder_path1
    labels_folder_path1 = args.labels_folder_path1
    point_cloud_folder_path2 = args.point_cloud_folder_path2
    labels_folder_path2 = args.labels_folder_path2

    point_cloud_folder_paths = [point_cloud_folder_path1, point_cloud_folder_path2]
    labels_folder_paths = [labels_folder_path1, labels_folder_path2]
    calculate_avg_points_within_objects(point_cloud_folder_paths, labels_folder_paths)
    # -- statistics R0_S3 ---
    # total cars: 472, total points inside cars: 28757, avg. points per cars: 60.925847
    # total trailers: 417, total points inside trailers: 202016, avg. points per trailers: 484.450839
    # total vans: 79, total points inside vans: 16663, avg. points per vans: 210.924051
    # total pedestrians: 200, total points inside pedestrians: 4216, avg. points per pedestrians: 21.080000
