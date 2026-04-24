# create a bar plot
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from internal.src.statistics.plot_utils import PlotUtils
import os


def log_tick_formatter(val, pos=None):
    if val < 1:
        return 0
    else:
        return r"$10^{{{}}}$".format(int(np.log10(val)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_folder_path_statistics", type=str, help="Path to output folder", default="")
    fontsize = 9
    args = argparser.parse_args()
    output_folder_path_statistics = args.output_folder_path_statistics
    if not os.path.exists(output_folder_path_statistics):
        os.makedirs(output_folder_path_statistics)

    plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Computer Modern Roman",
            }
        )

    labels = ['CAR', 'TRUCK', 'TRAILER', 'VAN', 'MOTORCYCLE', 'BUS', 'PEDESTRIAN', 'BICYCLE', 'SPECIAL_VEHICLE']
    class_colors = PlotUtils.get_class_colors(alpha=0.5)
    points = [32, 143, 524, 168, 0, 0, 21, 0, 0]
    x = np.arange(len(labels))
    # create 1200 x 750 pixels, 300 dpi plot
    fig = plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot(111)

    plt.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.41)
    plt.yticks(fontsize=fontsize)
    ax.bar(x, points)
    ax.set_ylabel('Number of 3D Points', fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize, ha='right')
    plt.xticks(rotation=45)
    plt.xlim(-0.5, len(labels)-0.5)
    ax.set_yscale("log")
    ax.set_yticks([0.8, 10 ** 1, 10 ** 2, 10 ** 3])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

    ax.yaxis.grid(True)
    ax.yaxis.grid(color='lightgray', alpha=0.5)

    # add y values on top of each bar
    for i in range(len(points)):
        plt.text(x[i], points[i]+points[i]/5 , str(points[i]), ha='center', fontsize=fontsize)

    for i in range(len(points)):
        ax.get_children()[i].set_color(class_colors[i])
    for i in range(len(points)):
        ax.get_children()[i].set_edgecolor('black')
        ax.get_children()[i].set_linewidth(1)

    avg_points = round(np.mean(points))
    plt.axhline(y=avg_points, color='r', linestyle='--', label='Average Points', linewidth=1)
    plt.text(5, avg_points+20, str(avg_points), ha='center', fontsize=fontsize, color='r')

    plt.show()
    fig.savefig(os.path.join(output_folder_path_statistics, 'avg_points_within_objects.pdf'))