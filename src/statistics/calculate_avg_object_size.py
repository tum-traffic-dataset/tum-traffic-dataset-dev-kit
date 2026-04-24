import argparse
import os
import json

if __name__ == "__main__":
    # add arg parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_folder_path_labels_train",
        type=str,
        help="Path to train labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_val",
        type=str,
        help="Path to val labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sampled",
        type=str,
        help="Path to test sampled labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sequence",
        type=str,
        help="Path to test sequence labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_south",
        type=str,
        help="Path to r01_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_north",
        type=str,
        help="Path to r02_s01 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_south",
        type=str,
        help="Path to r02_s02 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_north",
        type=str,
        help="Path to r02_s02 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_south",
        type=str,
        help="Path to r02_s03 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_north",
        type=str,
        help="Path to r02_s03 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_south",
        type=str,
        help="Path to r02_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_north",
        type=str,
        help="Path to r02_s04 north lidar labels",
        default="",
    )

    args = arg_parser.parse_args()

    input_folder_paths_all = []
    if args.input_folder_path_labels_train:
        input_folder_paths_all.append(args.input_folder_path_labels_train)
    if args.input_folder_path_labels_val:
        input_folder_paths_all.append(args.input_folder_path_labels_val)
    if args.input_folder_path_labels_test_sampled:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sampled)
    if args.input_folder_path_labels_test_sequence:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sequence)

    if args.input_folder_path_labels_sequence_s01_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_south)
    if args.input_folder_path_labels_sequence_s01_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_north)
    if args.input_folder_path_labels_sequence_s02_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_south)
    if args.input_folder_path_labels_sequence_s02_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_north)
    if args.input_folder_path_labels_sequence_s03_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_south)
    if args.input_folder_path_labels_sequence_s03_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_north)
    if args.input_folder_path_labels_sequence_s04_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_south)
    if args.input_folder_path_labels_sequence_s04_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_north)

    car_lengths_total = []
    car_widths_total = []
    car_heights_total = []
    truck_lengths_total = []
    truck_widths_total = []
    truck_heights_total = []
    trailer_lengths_total = []
    trailer_widths_total = []
    trailer_heights_total = []
    van_lengths_total = []
    van_widths_total = []
    van_heights_total = []
    motorcycle_lengths_total = []
    motorcycle_widths_total = []
    motorcycle_heights_total = []
    bus_lengths_total = []
    bus_widths_total = []
    bus_heights_total = []
    pedestrian_lengths_total = []
    pedestrian_widths_total = []
    pedestrian_heights_total = []
    bicycle_lengths_total = []
    bicycle_widths_total = []
    bicycle_heights_total = []
    emergency_vehicle_lengths_total = []
    emergency_vehicle_widths_total = []
    emergency_vehicle_heights_total = []
    other_lengths_total = []
    other_widths_total = []
    other_heights_total = []
    num_cars_total = 0
    num_trailers_total = 0
    num_trucks_total = 0
    num_vans_total = 0
    num_motorcycles_total = 0
    num_busses_total = 0
    num_pedestrians_total = 0
    num_bicycles_total = 0
    num_emergency_vehicles_total = 0
    num_others_total = 0
    for input_folder_path_labels in input_folder_paths_all:
        car_lengths = []
        car_widths = []
        car_heights = []
        truck_lengths = []
        truck_widths = []
        truck_heights = []
        trailer_lengths = []
        trailer_widths = []
        trailer_heights = []
        van_lengths = []
        van_widths = []
        van_heights = []
        motorcycle_lengths = []
        motorcycle_widths = []
        motorcycle_heights = []
        bus_lengths = []
        bus_widths = []
        bus_heights = []
        pedestrian_lengths = []
        pedestrian_widths = []
        pedestrian_heights = []
        bicycle_lengths = []
        bicycle_widths = []
        bicycle_heights = []
        emergency_vehicle_lengths = []
        emergency_vehicle_widths = []
        emergency_vehicle_heights = []
        other_lengths = []
        other_widths = []
        other_heights = []

        def append_data(length, width, height, category):
            if category == "CAR":
                car_lengths.append(length)
                car_widths.append(width)
                car_heights.append(height)
            elif category == "TRAILER":
                trailer_lengths.append(length)
                trailer_widths.append(width)
                trailer_heights.append(height)
            elif category == "TRUCK":
                truck_lengths.append(length)
                truck_widths.append(width)
                truck_heights.append(height)
            elif category == "VAN":
                van_lengths.append(length)
                van_widths.append(width)
                van_heights.append(height)
            elif category == "MOTORCYCLE":
                motorcycle_lengths.append(length)
                motorcycle_widths.append(width)
                motorcycle_heights.append(height)
            elif category == "BUS":
                bus_lengths.append(length)
                bus_widths.append(width)
                bus_heights.append(height)
            elif category == "PEDESTRIAN":
                pedestrian_lengths.append(length)
                pedestrian_widths.append(width)
                pedestrian_heights.append(height)
            elif category == "BICYCLE":
                bicycle_lengths.append(length)
                bicycle_widths.append(width)
                bicycle_heights.append(height)
            elif category == "EMERGENCY_VEHICLE":
                emergency_vehicle_lengths.append(length)
                emergency_vehicle_widths.append(width)
                emergency_vehicle_heights.append(height)
            elif category == "OTHER":
                other_lengths.append(length)
                other_widths.append(width)
                other_heights.append(height)

        for label_file_name in sorted(os.listdir(input_folder_path_labels)):
            json_file = open(
                os.path.join(input_folder_path_labels, label_file_name),
            )
            json_data = json.load(json_file)
            if "openlabel" in json_data:
                for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                    for object_track_id, object_json in frame_obj["objects"].items():
                        # Dataset in ASAM OpenLABEL format
                        length = float(object_json["object_data"]["cuboid"]["val"][7])
                        width = float(object_json["object_data"]["cuboid"]["val"][8])
                        height = float(object_json["object_data"]["cuboid"]["val"][9])
                        category = object_json["object_data"]["type"].upper()
                        append_data(length, width, height, category)
            else:
                for label in json_data["labels"]:
                    if "dimensions" in label:
                        # Dataset R1 NOT IN ASAM OpenLABEL format
                        length = float(label["dimensions"]["length"])
                        width = float(label["dimensions"]["width"])
                        height = float(label["dimensions"]["height"])
                    else:
                        # Dataset R0 NOT IN ASAM OpenLABEL format
                        length = float(label["box3d"]["dimension"]["length"])
                        width = float(label["box3d"]["dimension"]["width"])
                        height = float(label["box3d"]["dimension"]["height"])
                    category = label["category"].upper()
                    append_data(length, width, height, category)

        car_lengths_total.append(sum(car_lengths))
        car_widths_total.append(sum(car_widths))
        car_heights_total.append(sum(car_heights))
        trailer_lengths_total.append(sum(trailer_lengths))
        trailer_widths_total.append(sum(trailer_widths))
        trailer_heights_total.append(sum(trailer_heights))
        truck_lengths_total.append(sum(truck_lengths))
        truck_widths_total.append(sum(truck_widths))
        truck_heights_total.append(sum(truck_heights))
        van_lengths_total.append(sum(van_lengths))
        van_widths_total.append(sum(van_widths))
        van_heights_total.append(sum(van_heights))
        motorcycle_lengths_total.append(sum(motorcycle_lengths))
        motorcycle_widths_total.append(sum(motorcycle_widths))
        motorcycle_heights_total.append(sum(motorcycle_heights))
        bus_lengths_total.append(sum(bus_lengths))
        bus_widths_total.append(sum(bus_widths))
        bus_heights_total.append(sum(bus_heights))
        pedestrian_lengths_total.append(sum(pedestrian_lengths))
        pedestrian_widths_total.append(sum(pedestrian_widths))
        pedestrian_heights_total.append(sum(pedestrian_heights))
        bicycle_lengths_total.append(sum(bicycle_lengths))
        bicycle_widths_total.append(sum(bicycle_widths))
        bicycle_heights_total.append(sum(bicycle_heights))
        emergency_vehicle_lengths_total.append(sum(emergency_vehicle_lengths))
        emergency_vehicle_widths_total.append(sum(emergency_vehicle_widths))
        emergency_vehicle_heights_total.append(sum(emergency_vehicle_heights))
        other_lengths_total.append(sum(other_lengths))
        other_widths_total.append(sum(other_widths))
        other_heights_total.append(sum(other_heights))

        num_cars_total = num_cars_total + len(car_lengths)
        num_trailers_total = num_trailers_total + len(trailer_lengths)
        num_trucks_total = num_trucks_total + len(truck_lengths)
        num_vans_total = num_vans_total + len(van_lengths)
        num_motorcycles_total = num_motorcycles_total + len(motorcycle_lengths)
        num_busses_total = num_busses_total + len(bus_lengths)
        num_pedestrians_total = num_pedestrians_total + len(pedestrian_lengths)
        num_bicycles_total = num_bicycles_total + len(bicycle_lengths)
        num_emergency_vehicles_total = num_emergency_vehicles_total + len(emergency_vehicle_lengths)
        num_others_total = num_others_total + len(other_lengths)

        print(
            "num cars: %d, avg. length: %f, avg. width: %f, avg. height: %f"
            % (
                len(car_lengths),
                sum(car_lengths) / len(car_lengths),
                sum(car_widths) / len(car_widths),
                sum(car_heights) / len(car_heights),
            )
        )

        print(
            "num trailers: %d, avg. length: %f, avg. width: %f, avg. height: %f"
            % (
                len(trailer_lengths),
                sum(trailer_lengths) / len(trailer_lengths),
                sum(trailer_widths) / len(trailer_widths),
                sum(trailer_heights) / len(trailer_heights),
            )
        )

        if len(truck_lengths) > 0:
            print(
                "num trucks: %d, avg. length: %f, avg. width: %f, avg. height: %f"
                % (
                    len(truck_lengths),
                    sum(truck_lengths) / len(truck_lengths),
                    sum(truck_widths) / len(truck_widths),
                    sum(truck_heights) / len(truck_heights),
                )
            )

        print(
            "num vans: %d, avg. length: %f, avg. width: %f, avg. height: %f"
            % (
                len(van_lengths),
                sum(van_lengths) / len(van_lengths),
                sum(van_widths) / len(van_widths),
                sum(van_heights) / len(van_heights),
            )
        )

        if len(motorcycle_lengths) > 0:
            print(
                "num motorcycles: %d, avg. length: %f, avg. width: %f, avg. height: %f"
                % (
                    len(motorcycle_lengths),
                    sum(motorcycle_lengths) / len(motorcycle_lengths),
                    sum(motorcycle_widths) / len(motorcycle_widths),
                    sum(motorcycle_heights) / len(motorcycle_heights),
                )
            )

        if len(bus_lengths) > 0:
            print(
                "num buss: %d, avg. length: %f, avg. width: %f, avg. height: %f"
                % (
                    len(bus_lengths),
                    sum(bus_lengths) / len(bus_lengths),
                    sum(bus_widths) / len(bus_widths),
                    sum(bus_heights) / len(bus_heights),
                )
            )

        if len(pedestrian_lengths) > 0:
            print(
                "num pedestrians: %d, avg. length: %f, avg. width: %f, avg. height: %f"
                % (
                    len(pedestrian_lengths),
                    sum(pedestrian_lengths) / len(pedestrian_lengths),
                    sum(pedestrian_widths) / len(pedestrian_widths),
                    sum(pedestrian_heights) / len(pedestrian_heights),
                )
            )

        if len(bicycle_lengths) > 0:
            print(
                "num bicycles: %d, avg. length: %f, avg. width: %f, avg. height: %f"
                % (
                    len(bicycle_lengths),
                    sum(bicycle_lengths) / len(bicycle_lengths),
                    sum(bicycle_widths) / len(bicycle_widths),
                    sum(bicycle_heights) / len(bicycle_heights),
                )
            )

        if len(emergency_vehicle_lengths) > 0:
            print(
                "num emergency_vehicles: %d, avg. length: %f, avg. width: %f, avg. height: %f"
                % (
                    len(emergency_vehicle_lengths),
                    sum(emergency_vehicle_lengths) / len(emergency_vehicle_lengths),
                    sum(emergency_vehicle_widths) / len(emergency_vehicle_widths),
                    sum(emergency_vehicle_heights) / len(emergency_vehicle_heights),
                )
            )

        if len(other_lengths) > 0:
            print(
                "num others: %d, avg. length: %f, avg. width: %f, avg. height: %f"
                % (
                    len(other_lengths),
                    sum(other_lengths) / len(other_lengths),
                    sum(other_widths) / len(other_widths),
                    sum(other_heights) / len(other_heights),
                )
            )
        print("--------------------------------")

    print("-----statistics all sets (s1-s4)")

    print(
        "num cars total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_cars_total,
            sum(car_lengths_total) / num_cars_total,
            sum(car_widths_total) / num_cars_total,
            sum(car_heights_total) / num_cars_total,
        )
    )

    print(
        "num trailers total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_trailers_total,
            sum(trailer_lengths_total) / num_trailers_total,
            sum(trailer_widths_total) / num_trailers_total,
            sum(trailer_heights_total) / num_trailers_total,
        )
    )

    print(
        "num trucks total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_trucks_total,
            sum(truck_lengths_total) / num_trucks_total,
            sum(truck_widths_total) / num_trucks_total,
            sum(truck_heights_total) / num_trucks_total,
        )
    )

    print(
        "num vans total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_vans_total,
            sum(van_lengths_total) / num_vans_total,
            sum(van_widths_total) / num_vans_total,
            sum(van_heights_total) / num_vans_total,
        )
    )

    print(
        "num motorcycles total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_motorcycles_total,
            sum(motorcycle_lengths_total) / num_motorcycles_total,
            sum(motorcycle_widths_total) / num_motorcycles_total,
            sum(motorcycle_heights_total) / num_motorcycles_total,
        )
    )

    print(
        "num buss total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_busses_total,
            sum(bus_lengths_total) / num_busses_total,
            sum(bus_widths_total) / num_busses_total,
            sum(bus_heights_total) / num_busses_total,
        )
    )

    print(
        "num pedestrians total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_pedestrians_total,
            sum(pedestrian_lengths_total) / num_pedestrians_total,
            sum(pedestrian_widths_total) / num_pedestrians_total,
            sum(pedestrian_heights_total) / num_pedestrians_total,
        )
    )

    print(
        "num bicycles total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_bicycles_total,
            sum(bicycle_lengths_total) / num_bicycles_total,
            sum(bicycle_widths_total) / num_bicycles_total,
            sum(bicycle_heights_total) / num_bicycles_total,
        )
    )

    print(
        "num emergency_vehicles total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_emergency_vehicles_total,
            sum(emergency_vehicle_lengths_total) / num_emergency_vehicles_total,
            sum(emergency_vehicle_widths_total) / num_emergency_vehicles_total,
            sum(emergency_vehicle_heights_total) / num_emergency_vehicles_total,
        )
    )

    print(
        "num other total: %d, total avg. length: %f, total avg. width: %f, total avg. height: %f"
        % (
            num_others_total,
            sum(other_lengths_total) / num_others_total,
            sum(other_widths_total) / num_others_total,
            sum(other_heights_total) / num_others_total,
        )
    )

    print("--------------------------------")
