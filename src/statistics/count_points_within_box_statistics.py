import numpy as np
import open3d as o3d
import json
import argparse
import os


# Example:
# python count_points_within_box_statistics.py -p "/home/walter/Downloads/a9_r0_dataset/r0_s3/03_point_clouds/" -l "/home/walter/Downloads/a9_r0_dataset/r0_s3/04_labels/"


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
    car_points = 0
    trailer_points = 0
    truck_points = 0
    van_points = 0
    motorcycle_points = 0
    bus_points = 0
    pedestrian_points = 0
    bicycle_points = 0
    special_vehicle_points = 0

    num_cars = 0
    num_trailers = 0
    num_trucks = 0
    num_vans = 0
    num_motorcycles = 0
    num_busses = 0
    num_pedestrians = 0
    num_bicycles = 0
    num_special_vehicles = 0

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
                # label_idx = label_idx + 1
                if label["category"] == "CAR":
                    car_points = car_points + num_points_inside_box
                    num_cars = num_cars + 1
                elif label["category"] == "TRAILER":
                    trailer_points = trailer_points + num_points_inside_box
                    num_trailers = num_trailers + 1
                elif label["category"] == "TRUCK":
                    truck_points = truck_points + num_points_inside_box
                    num_trucks = num_trucks + 1
                elif label["category"] == "VAN":
                    van_points = van_points + num_points_inside_box
                    num_vans = num_vans + 1
                elif label["category"] == "MOTORCYCLE":
                    motorcycle_points = motorcycle_points + num_points_inside_box
                    num_motorcycles = num_motorcycles + 1
                elif label["category"] == "BUS":
                    bus_points = bus_points + num_points_inside_box
                    num_busses = num_busses + 1
                elif label["category"] == "PEDESTRIAN":
                    pedestrian_points = pedestrian_points + num_points_inside_box
                    num_pedestrians = num_pedestrians + 1
                elif label["category"] == "BICYCLE":
                    bicycle_points = bicycle_points + num_points_inside_box
                    num_pedestrians = num_pedestrians + 1
                elif label["category"] == "SPECIAL_VEHICLE":
                    special_vehicle_points = special_vehicle_points + num_points_inside_box
                    num_special_vehicles = num_special_vehicles + 1

            file_idx = file_idx + 1

        print("--statistics sub set---")
        print_statistic(bicycle_points, bus_points, car_points, motorcycle_points, num_bicycles, num_busses, num_cars,
                        num_motorcycles, num_pedestrians, num_special_vehicles, num_trailers, num_trucks, num_vans,
                        pedestrian_points, special_vehicle_points, trailer_points, truck_points, van_points)
    print("--statistics all sets---")
    print_statistic(bicycle_points, bus_points, car_points, motorcycle_points, num_bicycles, num_busses, num_cars,
                    num_motorcycles, num_pedestrians, num_special_vehicles, num_trailers, num_trucks, num_vans,
                    pedestrian_points, special_vehicle_points, trailer_points, truck_points, van_points)
    return [car_points, truck_points, trailer_points, van_points, motorcycle_points, bus_points, pedestrian_points,
            bicycle_points, special_vehicle_points]


def print_statistic(bicycle_points, bus_points, car_points, motorcycle_points, num_bicycles, num_busses, num_cars,
                    num_motorcycles, num_pedestrians, num_special_vehicles, num_trailers, num_trucks, num_vans,
                    pedestrian_points, special_vehicle_points, trailer_points, truck_points, van_points):
    if num_cars > 0:
        print("total cars: %d, total points inside cars: %d, avg. points per cars: %f" % (
            num_cars, car_points, car_points / num_cars))
    if num_trailers > 0:
        print("total trailers: %d, total points inside trailers: %d, avg. points per trailers: %f" % (
            num_trailers, trailer_points, trailer_points / num_trailers))
    if num_trucks > 0:
        print("total trucks: %d, total points inside trucks: %d, avg. points per trucks: %f" % (
            num_trucks, truck_points, truck_points / num_trucks))
    if num_vans > 0:
        print("total vans: %d, total points inside vans: %d, avg. points per vans: %f" % (
            num_vans, van_points, van_points / num_vans))
    if num_motorcycles > 0:
        print("total motorcycles: %d, total points inside motorcycles: %d, avg. points per motorcycles: %f" % (
            num_motorcycles, motorcycle_points, motorcycle_points / num_motorcycles))
    if num_busses > 0:
        print("total busses: %d, total points inside busses: %d, avg. points per bus: %f" % (
            num_busses, bus_points, bus_points / num_busses))
    if num_pedestrians > 0:
        print("total pedestrians: %d, total points inside pedestrians: %d, avg. points per pedestrians: %f" % (
            num_pedestrians, pedestrian_points, pedestrian_points / num_pedestrians))
    if num_bicycles > 0:
        print("total bicycles: %d, total points inside bicycles: %d, avg. points per bicycles: %f" % (
            num_bicycles, bicycle_points, bicycle_points / num_bicycles))
    if num_special_vehicles > 0:
        print(
            "total special_vehicles: %d, total points inside special_vehicles: %d, avg. points per special_vehicles: %f" % (
                num_special_vehicles, special_vehicle_points, special_vehicle_points / num_special_vehicles))


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
    argparser.add_argument('--output_folder_path_statistics', default="output",
                            help='Folder path of statistics. Default: output')

    args = argparser.parse_args()
    point_cloud_folder_path1 = args.point_cloud_folder_path1
    labels_folder_path1 = args.labels_folder_path1
    point_cloud_folder_path2 = args.point_cloud_folder_path2
    labels_folder_path2 = args.labels_folder_path2
    output_folder_path_statistics = args.output_folder_path_statistics
    if not os.path.exists(output_folder_path_statistics):
        os.makedirs(output_folder_path_statistics)

    point_cloud_folder_paths = [point_cloud_folder_path1, point_cloud_folder_path2]
    labels_folder_paths = [labels_folder_path1, labels_folder_path2]
    calculate_avg_points_within_objects(point_cloud_folder_paths, labels_folder_paths)
    # -- statistics R0_S3 ---
    # total cars: 472, total points inside cars: 28757, avg. points per cars: 60.925847
    # total trailers: 417, total points inside trailers: 202016, avg. points per trailers: 484.450839
    # total vans: 79, total points inside vans: 16663, avg. points per vans: 210.924051
    # total pedestrians: 200, total points inside pedestrians: 4216, avg. points per pedestrians: 21.080000
