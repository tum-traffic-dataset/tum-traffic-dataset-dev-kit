import numpy as np


def transform_base_to_lidar(boxes):
    # transforms data from base to south lidar
    transformation_matrix_s110_base_to_lidar_south = np.array(
        [
            [0.21479487, 0.97627129, -0.02752358, 1.36963590],
            [-0.97610281, 0.21553834, 0.02768645, -16.19616413],
            [0.03296187, 0.02091894, 0.99923766, -6.99999999],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ]
    )
    boxes_homogeneous = np.hstack((boxes, np.ones((boxes.shape[0], 1))))
    # transform boxes from s110 base frame to lidar frame
    boxes_transformed = np.dot(transformation_matrix_s110_base_to_lidar_south, boxes_homogeneous.T).T
    return boxes_transformed[:, :3]


def transform_lidar_into_base(points):
    # transform data from south lidar to base
    transformation_matrix_lidar_south_to_s110_base = np.array(
        [
            [0.21479485, -0.9761028, 0.03296187, -15.87257873],
            [0.97627128, 0.21553835, 0.02091894, 2.30019086],
            [-0.02752358, 0.02768645, 0.99923767, 7.48077521],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ]
    )
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # transform point cloud into s110 base frame
    points_transformed = np.dot(transformation_matrix_lidar_south_to_s110_base, points_homogeneous.T).T
    # set z coordinate to zero
    points_transformed[:, 2] = 0.0
    return points_transformed[:, :3]
