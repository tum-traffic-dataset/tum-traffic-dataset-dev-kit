import argparse
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

transformation_matrices = {
    "transformation_common_road_to_s040_camera_basler_north_16mm": {
        "src": "common_road",
        "dst": "s040_camera_basler_north_16mm",
        "transform_src_to_dst": {
            "matrix4x4": [1.62812820e-01, 9.86510190e-01, -1.70187000e-02, -5.95094598e+01,
                          2.15920790e-01, -5.24555400e-02, -9.75000830e-01, -9.09916911e+01,
                          -9.62740980e-01, 1.55067950e-01, -2.21548500e-01, 4.43080190e+02,
                          0.0, 0.0, 0.0, 1.0]
        }
    },
    "transformation_common_road_to_s040_camera_basler_north_50mm": {
        "src": "common_road",
        "dst": "s040_camera_basler_north_50mm",
        "transform_src_to_dst": {
            "matrix4x4": [7.54789063e-02, 9.97141308e-01, 3.48514044e-03, -2.06169874e+01,
                          5.62727716e-02, -7.69991457e-04, -9.98415135e-01, -1.79818775e+01,
                          -9.95558291e-01, 7.55554009e-02, -5.61700232e-02, 4.65221911e+02,
                          0.0, 0.0, 0.0, 1.0],
        }
    },
    "transformation_common_road_to_s050_camera_basler_south_16mm": {
        "src": "common_road",
        "dst": "s050_camera_basler_south_16mm",
        "transform_src_to_dst": {
            "matrix4x4": [-6.67942916e-04, -9.99986586e-01, -5.13624571e-03, -5.54729606e+00, -1.94233505e-01,
                          5.16816386e-03, -9.80941709e-01, 8.46243832e+00, 9.80955096e-01, 3.42417940e-04,
                          -1.94234351e-01, 8.28553587e-01, 0.0, 0.0, 0.0, 1.0],
        }
    },
    "transformation_common_road_to_s050_camera_basler_south_50mm": {
        "src": "common_road",
        "dst": "s050_camera_basler_south_50mm",
        "transform_src_to_dst": {
            "matrix4x4": [7.01156300e-02, -9.97500830e-01, -8.71162000e-03, -5.17308492e+00, -5.89034100e-02,
                          4.57780000e-03, -9.98253190e-01, 8.29447420e+00, 9.95798270e-01, 7.05063000e-02,
                          -5.84352200e-02, 4.41625452e-02, 0.0, 0.0, 0.0, 1.0],
        }
    }
}


def get_cuboid_corners(l, w, h):
    # Create a bounding box outline
    bounding_box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )

    # add half of the height to the z coordinate
    bounding_box[2, :] += h / 2

    translation = cuboid[:3]
    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    rotation_quaternion = cuboid[3:7]
    rotation_matrix = R.from_quat(rotation_quaternion).as_matrix()
    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
    return corner_box.transpose()


def get_transformation_matrix_by_sensor_id(sensor_id):
    for transformation_matrix in transformation_matrices.values():
        if transformation_matrix["dst"] == sensor_id:
            return np.array(transformation_matrix["transform_src_to_dst"]["matrix4x4"]).reshape(4, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path_boxes",
        type=str,
        help="Input directory path",
        default="path_to_boxes",
    )
    parser.add_argument(
        "--output_folder_path_boxes_transformed",
        type=str,
        help="Output directory path",
        default="path_to_output",
    )
    parser.add_argument(
        "--sensor_id",
        type=str,
        help="Sensor ID",
        default="s040_camera_basler_north_16mm",
    )

    args = parser.parse_args()
    input_folder_path_boxes = args.input_folder_path_boxes
    output_folder_path_boxes_transformed = args.output_folder_path_boxes_transformed
    sensor_id = args.sensor_id

    if not os.path.exists(output_folder_path_boxes_transformed):
        os.makedirs(output_folder_path_boxes_transformed)

    # transformation_common_road_to_camera_s40_near = np.array(
    #     [[0.16281282, 0.21592079, -0.96274098, -455.90735878],
    #      [0.98651019, -0.05245554, 0.15506795, 14.77386387],
    #      [-0.0170187, -0.97500083, -0.2215485, -8.434],
    #      [0.0, 0.0, 0.0, 1.0]])

    for file_name in sorted(os.listdir(args.input_folder_path_boxes)):
        label_data = json.load(open(os.path.join(args.input_folder_path_boxes, file_name)))
        valid_labels = {}
        frame_idx = -1
        detections = []
        for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
            frame_idx = frame_id
            for object_id, label in frame_obj["objects"].items():
                cuboid = np.array(label["object_data"]["cuboid"]["val"])
                # transform cuboid from common_road to camera space
                location_common_road = np.array([cuboid[0], cuboid[1], cuboid[2]])
                # add half of the height to the z coordinate
                location_common_road[2] += cuboid[9] / 2

                corners_3d = get_cuboid_corners(cuboid[7], cuboid[8], cuboid[9])  # 8x3
                # make corners_3d homogenous by adding 1
                corners_3d = np.concatenate([corners_3d, np.ones((8, 1))], axis=1).T
                transformation_common_road_to_camera = get_transformation_matrix_by_sensor_id(sensor_id)
                corners_3d_in_camera = np.matmul(transformation_common_road_to_camera, corners_3d)
                print(corners_3d_in_camera)

                rotation_common_road_quaternion = np.array([cuboid[3], cuboid[4], cuboid[5], cuboid[6]])
                rotation_matrix_common_road = R.from_quat(rotation_common_road_quaternion).as_matrix()
                object_pose_in_common_road = np.eye(4)
                object_pose_in_common_road[:3, :3] = rotation_matrix_common_road
                object_pose_in_common_road[:3, 3] = location_common_road[:3]

                object_pose_in_camera = np.matmul(transformation_common_road_to_camera,
                                                  object_pose_in_common_road)
                # store back the location in the cuboid
                label["object_data"]["cuboid"]["val"][0] = object_pose_in_camera[0, 3]
                label["object_data"]["cuboid"]["val"][1] = object_pose_in_camera[1, 3]
                label["object_data"]["cuboid"]["val"][2] = object_pose_in_camera[2, 3]

                rotation_matrix_in_camera = object_pose_in_camera[:3, :3]
                rotation_quaternion_in_camera = R.from_matrix(rotation_matrix_in_camera).as_quat()
                label["object_data"]["cuboid"]["val"][3] = rotation_quaternion_in_camera[0]
                label["object_data"]["cuboid"]["val"][4] = rotation_quaternion_in_camera[1]
                label["object_data"]["cuboid"]["val"][5] = rotation_quaternion_in_camera[2]
                label["object_data"]["cuboid"]["val"][6] = rotation_quaternion_in_camera[3]

        # save undistorted bounding boxes to json file
        with open(os.path.join(output_folder_path_boxes_transformed, file_name), "w") as f:
            json.dump(label_data, f, indent=4, sort_keys=True)
