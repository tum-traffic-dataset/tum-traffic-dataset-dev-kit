import numpy as np
import open3d as o3d
import argparse
import os

# This script visualizes 3D labels in a point cloud scan.
# Usage:
#           python a9-dataset-dev_kit/visualize_point_cloud_with_labels.py --file_path_point_cloud <FILE_PATH_POINT_CLOUD> \
#                                                                  --file_path_labels <FILE_PATH_LABELS>
# Example:
#           python a9-dataset-dev_kit/visualize_point_cloud_with_labels.py --file_path_point_cloud input/point_cloud.pcd \
#                                                                  --file_path_labels input/labels.json
from pathlib import Path
import sys
import progressbar

sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from src.utils.vis_utils import VisualizationUtils


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="VizLabel Argument Parser")
    argparser.add_argument(
        "--input_folder_path_point_clouds",
        default="input/point_cloud",
        help="Point cloud file path. Default: input/point_cloud.pcd",
    )
    argparser.add_argument("--input_folder_path_detections", help="Folder path to detections (in OpenLABEL format).")
    argparser.add_argument("--input_folder_path_labels", help="Folder path to labels (in OpenLABEL format).")
    argparser.add_argument(
        "--save_visualization_results", action="store_true", help="Flag whether to save visualization results."
    )
    argparser.add_argument(
        "--show_visualization_results", action="store_true", help="Flag whether to show visualization results."
    )
    argparser.add_argument(
        "--output_folder_path_visualization_results",
        default="output/visualization_3d/lidar/r01_s04/s110_lidar_ouster_south/",
        help="Output folder path visualization results.",
    )
    argparser.add_argument(
        "--view",
        default="wide",
        help="View to the point cloud. Possible values are: [sensor, bev, wide, custom, s110_camera_basler_south2]. Default: wide",
    )
    argparser.add_argument(
        "--use_detections_in_base", action="store_true", help="Use detections in base coordinate system."
    )
    # parse width and height
    argparser.add_argument("--rendering_width", default=1920, help="Width of the window. Default: 1920")
    argparser.add_argument("--rendering_height", default=1080, help="Height of the window. Default: 1080")
    args = argparser.parse_args()
    input_folder_path_point_clouds = args.input_folder_path_point_clouds
    input_folder_path_detections = args.input_folder_path_detections
    input_folder_path_labels = args.input_folder_path_labels
    save_visualization_results = args.save_visualization_results
    show_visualization_results = args.show_visualization_results
    output_folder_path_visualization_results = args.output_folder_path_visualization_results
    view = args.view

    vis_utils = VisualizationUtils()
    renderer = o3d.visualization.rendering.OffscreenRenderer(args.rendering_width, args.rendering_height)
    renderer.scene.set_background([0.2, 0.2, 0.2, 1.0])
    renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.point_size = 4.0
    mat.line_width = 2.0  # 10.0

    if not os.path.exists(output_folder_path_visualization_results):
        Path(output_folder_path_visualization_results).mkdir(parents=True, exist_ok=True)

    bar = progressbar.ProgressBar(maxval=len(os.listdir(input_folder_path_point_clouds))).start()
    file_idx = 0
    point_cloud_file_names = sorted(os.listdir(input_folder_path_point_clouds))

    if input_folder_path_labels is not None:
        label_file_names = sorted(os.listdir(input_folder_path_labels))
    else:
        label_file_names = [""] * len(point_cloud_file_names)

    if input_folder_path_detections is not None:
        detection_file_names = sorted(os.listdir(input_folder_path_detections))
    else:
        detection_file_names = [""] * len(point_cloud_file_names)

    for point_cloud_file_name, label_file_name, detection_file_name in zip(
        point_cloud_file_names, label_file_names, detection_file_names
    ):
        print("processing file: ", point_cloud_file_name)
        file_path_point_cloud = os.path.join(input_folder_path_point_clouds, point_cloud_file_name)
        if input_folder_path_labels is not None:
            file_path_labels = os.path.join(input_folder_path_labels, label_file_name)
        else:
            file_path_labels = ""

        if input_folder_path_detections is not None:
            file_path_detections = os.path.join(input_folder_path_detections, detection_file_name)
        else:
            file_path_detections = ""

        vis_utils.visualize_boxes_3d(
            file_path_point_cloud,
            file_path_labels,
            file_path_detections,
            view,
            use_detections_in_base=bool(args.use_detections_in_base),
            save_visualization_results=save_visualization_results,
            show_visualization_results=show_visualization_results,
            output_folder_path_visualization_results=output_folder_path_visualization_results,
            renderer=renderer,
            material=mat,
        )
        bar.update(file_idx)
        file_idx = file_idx + 1
