import os
import argparse
import json
import numpy as np


def find_rosbag_with_most_scenarios(input_file_path_scenarios):
    rosbag_scenarios_all = json.load(open(os.path.join(input_file_path_scenarios), ))
    max_speed = 0
    max_speed_idx = -1
    for idx, date in enumerate(sorted(rosbag_scenarios_all.keys())):
        rosbag_scenarios = rosbag_scenarios_all[date]
        if float(rosbag_scenarios["top_speed"]) > max_speed:
            max_speed = float(rosbag_scenarios["top_speed"])
            max_speed_idx = idx
    print("max speed: ", str(max_speed))
    print("idx with max speed: ", str(max_speed_idx))
    rosbag_scenarios_all_list = list(rosbag_scenarios_all)
    print("date of rosbag with max speed: ", str(rosbag_scenarios_all_list[max_speed_idx]))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Argument Parser')
    argparser.add_argument('-i', '--input_file_path_scenarios', default="rosbag_scenarios.json",
                           help='input file path of scenarios file (json)')
    args = argparser.parse_args()
    input_file_path_scenarios = args.input_file_path_scenarios
    find_rosbag_with_most_scenarios(input_file_path_scenarios)
