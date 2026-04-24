import os
import argparse
import json
import numpy as np


def find_rosbag_with_most_scenarios(input_file_path_scenarios):
    rosbag_scenarios_all = json.load(open(os.path.join(input_file_path_scenarios), ))
    num_scenarios = []
    for date in sorted(rosbag_scenarios_all.keys()):
        num_scenarios.append([date, sum(rosbag_scenarios_all[date].values())])
    keys = [item[0] for item in num_scenarios]
    values = [item[1] for item in num_scenarios]
    max_item = max(values)
    print(values)
    idx = values.index(max_item)
    print(num_scenarios)
    print("idx of rosbag with most scenarios: ", str(idx))
    print("date of rosbag with most scenarios: ", str(keys[idx]))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Argument Parser')
    argparser.add_argument('-i', '--input_file_path_scenarios', default="rosbag_scenarios.json",
                           help='input file path of scenarios file (json)')
    args = argparser.parse_args()
    input_file_path_scenarios = args.input_file_path_scenarios
    find_rosbag_with_most_scenarios(input_file_path_scenarios)
