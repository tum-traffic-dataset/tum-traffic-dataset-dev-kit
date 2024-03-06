import cv2
import numpy as np
import open3d as o3d
import argparse
import os

# This script visualizes 3D labels in a point cloud scan.
# Usage:
#           python a9-dataset-dev_kit/visualize_point_cloud_with_3d_boxes.py --file_path_point_cloud <FILE_PATH_POINT_CLOUD> \
#                                                                  --file_path_labels <FILE_PATH_LABELS>
# Example:
#           python a9-dataset-dev_kit/visualize_point_cloud_with_3d_boxes.py --file_path_point_cloud input/point_cloud.pcd \
#                                                                  --file_path_labels input/labels.json
from pathlib import Path
import sys
import progressbar
from open3d.visualization import rendering
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from src.utils.vis_utils import VisualizationUtils
import src.map.hd_map as hdmap


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
        help="View to the point cloud. Possible values are: [bev, wide, custom, s110_camera_basler_south2]. Default: wide",
    )
    argparser.add_argument(
        "--use_detections_in_base", action="store_true", help="Use detections in base coordinate system."
    )
    # parse width and height
    argparser.add_argument("--rendering_width", default=1920, help="Width of the window. Default: 1920")
    argparser.add_argument("--rendering_height", default=1200, help="Height of the window. Default: 1080")
    argparser.add_argument("--show_hd_map", action="store_true", help="Flag whether to show HD map.")
    args = argparser.parse_args()
    input_folder_path_point_clouds = args.input_folder_path_point_clouds
    input_folder_path_detections = args.input_folder_path_detections
    input_folder_path_labels = args.input_folder_path_labels
    save_visualization_results = args.save_visualization_results
    show_visualization_results = args.show_visualization_results
    output_folder_path_visualization_results = args.output_folder_path_visualization_results
    view = args.view
    use_detections_in_base = args.use_detections_in_base
    show_hd_map = args.show_hd_map

    vis_utils = VisualizationUtils()

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.rendering_width, args.rendering_height)
    renderer.scene.set_background([0.2, 0.2, 0.2, 1.0])
    renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))
    point_material = o3d.visualization.rendering.MaterialRecord()
    point_material.point_size = 4.0
    # Some platforms do not require OpenGL implementations to support wide lines,
    # so the renderer requires a custom shader to implement this: "unlitLine".
    # The line_width field is only used by this shader; all other shaders ignore
    # it.
    line_material = o3d.visualization.rendering.MaterialRecord()
    line_material.shader = "litLine"
    line_material.line_width = 5.0  # changing line width size has no effect
    material = {
        "point": point_material,
        "line": line_material,
    }

    lane_sections = hdmap.load_map_for_local_frame("s110_base")
    # filter lane section to 200 m x 200 m region around s110 base
    lane_sections_filtered = []
    for lane_section in lane_sections:
        lane_section = lane_section.crop_to_area(min_pos=np.array([-300, -300]), max_pos=np.array([300, 300]))
        if lane_section:
            lane_sections_filtered.append(lane_section)

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

        pcd = o3d.io.read_point_cloud(file_path_point_cloud)
        object_id_list = vis_utils.visualize_boxes_3d(
            pcd,
            file_path_point_cloud,
            file_path_labels,
            file_path_detections,
            use_detections_in_base=bool(args.use_detections_in_base),
            renderer=renderer,
            material=material,
        )

        """
        Parameters:
        fov:            vertical field of view
        lookat_vector:  [center, eye, up]
                         center describes the point the camera is looking at.
                         eye describes the position of the camera.
                         up describes the up direction of the camera.
        """
        if view == "bev":
            renderer.scene.camera.set_projection(rendering.Camera.Projection.Ortho, -120, 120, -60, 60, 0.0, 100.0)
            center = np.array([30, 0, 0])
            eye = np.array([30, 0, 1])
            up_vector = np.array([0, 0, 1])
            lookat_vector = np.array([center, eye, up_vector])
            renderer.scene.camera.look_at(lookat_vector[0], lookat_vector[1], lookat_vector[2])
        elif view == "wide":
            center = np.array([17.439, 4.825, -2.311])  # = lookat
            eye = np.array([-0.945, 0.039, 0.325])  # = front
            up_vector = np.array([0.325, -0.010, 0.946])  # up
            lookat_vector = np.array([center, eye, up_vector])
            renderer.setup_camera(120, lookat_vector[0], lookat_vector[1], lookat_vector[2])
        elif view == "custom":
            #  center, eye, up
            lookat_vector = np.array([[5.0, 2.0, 0.0], [0.0, 2.0, 10.0], [1.0, 0.0, 0.0]])
            renderer.setup_camera(120, lookat_vector[0], lookat_vector[1], lookat_vector[2])
        else:
            raise ValueError("Unknown view type {}".format(view))

        if show_hd_map:
            # plot all lane sections from hd map
            lane_positions = []
            line_indices = []
            translation = [-15.87257873, 2.30019086, 7.48077521]
            for lane_section in lane_sections_filtered:
                for lane_positions_np_arr in lane_section.lanes:
                    if len(lane_positions_np_arr) > 1:
                        for idx, lane_position in enumerate(lane_positions_np_arr):
                            # transform position from s110_base to lidar frame
                            lane_pos = np.array(lane_position) - np.array(translation)
                            lane_positions.append([lane_pos[0], lane_pos[1], lane_pos[2]])
                            if idx > 0:
                                line_indices.append([len(lane_positions) - 2, len(lane_positions) - 1])

            # rotate lane position by 90 degrees around z axis
            lane_positions = np.array(lane_positions)
            rotation_z = 77.8
            rotation_z = rotation_z * np.pi / 180.0
            lane_positions = np.dot(lane_positions, np.array(
                [[np.cos(rotation_z), -np.sin(rotation_z), 0], [np.sin(rotation_z), np.cos(rotation_z), 0], [0, 0, 1]]))

            line_set = o3d.geometry.LineSet()
            color = (0.15, 0.15, 0.15)
            # color = (1, 1, 1)
            colors = [color for i in range(len(line_indices))]
            line_set.points = o3d.utility.Vector3dVector(lane_positions)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            renderer.scene.add_geometry("map_line_set", line_set, material["line"])

        if save_visualization_results:
            image_o3d = renderer.render_to_image()
            image_cv2 = np.array(image_o3d)
            file_name = os.path.basename(file_path_point_cloud)
            file_name_without_extension = os.path.splitext(file_name)[0]
            #output_file_path = os.path.join(output_folder_path_visualization_results, file_name.replace(".pcd", ".jpg"))
            # TODO: make it dynamic (automatically load .bin or .pcd)
            # TODO: .bin point clouds are not yet visualized -> fix bug
            output_file_path = os.path.join(output_folder_path_visualization_results, file_name.replace(".bin", ".jpg"))
            print("Saving visualization results to {}".format(output_file_path))
            cv2.imwrite(output_file_path, image_cv2)

        if show_visualization_results:
            # center, eye, up
            o3d.visualization.draw_geometries(
                [pcd], zoom=0.3412, front=lookat_vector[0], lookat=lookat_vector[1], up=lookat_vector[2]
            )

        renderer.scene.remove_geometry("pcd")
        # iterate all object id and remove geometry
        for object_id in object_id_list:
            renderer.scene.remove_geometry("line_set_" + object_id)
            renderer.scene.remove_geometry("track_history_line_set_" + object_id)

        bar.update(file_idx)
        file_idx = file_idx + 1
    vis_utils = None
