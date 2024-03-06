import os
import sys
from pathlib import Path
import numpy as np
from src.utils.point_cloud_registration_utils import (
    read_point_cloud_with_intensity,
    parse_parameters,
    write_point_cloud_with_intensity,
    filter_point_cloud,
    normalize_intensity,
)

np.set_printoptions(suppress=True)
import open3d as o3d


# Register source point cloud to target point cloud and find the optimal transformation among them.
# The final result is stored in the target coordinate system.

# Example: python point_cloud_registration.py --folder_path_point_cloud_source /home/user/Downloads/test_frames/lidar_vehicle_point_cloud/ --folder_path_point_cloud_target /home/user/Downloads/test_frames/lidar_infrastructure_point_cloud/ --initial_voxel_size 2 --continuous_voxel_size 2 --output_folder_path_registered_point_clouds /home/user/Downloads/test_frames/registered_point_clouds/ --save_registered_point_clouds
# NOTE: set the initial transformation matrix in line 195!


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return pcd_down, pcd_fpfh


def prepare_point_cloud(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_initial_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(source, target, voxel_size, trans_init):
    use_point_to_plane = False
    distance_threshold = voxel_size * 0.4
    if use_point_to_plane:
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
    else:
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
    return result


def get_xyzi_points(cloud_array, remove_nans=True, dtype=float):
    if remove_nans:
        mask = np.isfinite(cloud_array["x"]) & np.isfinite(cloud_array["y"]) & np.isfinite(cloud_array["z"])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[..., 0] = cloud_array["x"]
    points[..., 1] = cloud_array["y"]
    points[..., 2] = cloud_array["z"]
    points[..., 3] = cloud_array["intensity"]

    return points


def register_point_clouds(
    do_initial_registration,
    point_cloud_source,
    point_cloud_target,
    idx_file,
    transformation_matrix,
    initial_voxel_size,
    continuous_voxel_size,
):
    num_initial_registration_loops = 4

    inlier_rmse_best = sys.maxsize
    fitness_best = 0
    loss_best = sys.maxsize

    point_cloud_array_source = np.array(point_cloud_source)
    point_cloud_array_target = np.array(point_cloud_target)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point_cloud_array_source[:, 0:3])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(point_cloud_array_target[:, 0:3])

    voxel_size = continuous_voxel_size
    if do_initial_registration:
        for i in range(num_initial_registration_loops):
            # NOTE: the smaller the voxel_size the more accurate is the initial transformation matrix
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_point_cloud(
                source, target, initial_voxel_size
            )
            initial_registration_result = execute_initial_registration(
                source_down, target_down, source_fpfh, target_fpfh, initial_voxel_size
            )

            loss = initial_registration_result.inlier_rmse + 1 / initial_registration_result.fitness
            if loss < loss_best:
                transformation_matrix = initial_registration_result.transformation
                inlier_rmse_best = initial_registration_result.inlier_rmse
                fitness_best = initial_registration_result.fitness
                loss_best = loss
                print("==========================")
                print(
                    "Better transformation matrix found (using initial registration):\n ", repr(transformation_matrix)
                )
                print("With frame index: %d" % idx_file)
                print("With better RMSE: %.4f" % inlier_rmse_best)
                print("With better fitness: %.4f" % fitness_best)
                print("With better loss: %.4f" % loss_best)
                print("==========================")

            if transformation_matrix is not None:
                continuous_registration_result = refine_registration(
                    source_down, target_down, voxel_size, transformation_matrix
                )
                if (
                    continuous_registration_result.inlier_rmse < inlier_rmse_best
                    and continuous_registration_result.fitness > fitness_best
                ):
                    inlier_rmse_best = continuous_registration_result.inlier_rmse
                    fitness_best = continuous_registration_result.fitness
                    transformation_matrix = continuous_registration_result.transformation
                    print("==========================")
                    print(
                        "Better transformation matrix found (using continuous registration):\n ",
                        repr(transformation_matrix),
                    )
                    print("With frame index: %d" % idx_file)
                    print("With better RMSE: %.4f" % inlier_rmse_best)
                    print("With better fitness: %.4f" % fitness_best)
                    print("==========================")
    else:
        print("Initial transformation_matrix: \n", repr(transformation_matrix))

    if transformation_matrix is not None:
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        continuous_registration_result = refine_registration(
            source_down, target_down, voxel_size, transformation_matrix
        )
        loss = continuous_registration_result.inlier_rmse + 1 / continuous_registration_result.fitness
        if loss < loss_best:
            inlier_rmse_best = continuous_registration_result.inlier_rmse
            fitness_best = continuous_registration_result.fitness
            loss_best = loss
            transformation_matrix = continuous_registration_result.transformation
            print("==========================")
            print("Better transformation matrix found (using continuous registration): \n", repr(transformation_matrix))
            print("With frame index: %d" % idx_file)
            print("With better RMSE: %.4f" % inlier_rmse_best)
            print("With better fitness: %.4f" % fitness_best)
            print("With better loss: %.4f" % loss_best)
            print("==========================")
    return transformation_matrix, inlier_rmse_best, fitness_best


if __name__ == "__main__":
    args = parse_parameters()

    if not os.path.exists(args.output_folder_path_registered_point_clouds):
        os.makedirs(args.output_folder_path_registered_point_clouds)

    point_cloud_source_file_names = sorted(os.listdir(args.folder_path_point_cloud_source))
    point_cloud_target_file_names = sorted(os.listdir(args.folder_path_point_cloud_target))

    idx_file = 0
    do_initial_registration = False
    transformation_matrix_best = None
    inlier_rmse_global_best = sys.maxsize
    fitness_global_best = 0.0
    loss_best = sys.maxsize
    if (
        "s110_lidar_ouster_north" in args.folder_path_point_cloud_source
        and "s110_lidar_ouster_south" in args.folder_path_point_cloud_target
    ):
        transformation_matrix_initial = np.array(
            [
                [0.95812397, -0.28634765, -0.00186487, 1.39543971],
                [0.28633102, 0.95810969, -0.00635344, -13.9356065],
                [0.00360605, 0.00555341, 0.99997808, 0.1595251],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif (
        "s110_lidar_ouster_south" in args.folder_path_point_cloud_source
        and "s110_lidar_ouster_north" in args.folder_path_point_cloud_target
    ):
        transformation_matrix_initial = np.linalg.inv(
            np.array(
                [
                    [0.95812397, -0.28634765, -0.00186487, 1.39543971],
                    [0.28633102, 0.95810969, -0.00635344, -13.9356065],
                    [0.00360605, 0.00555341, 0.99997808, 0.1595251],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    else:
        print("ERROR: unknown lidar pair. Supported LiDAR IDs are: [s110_lidar_ouster_south, s110_lidar_ouster_north]")
        exit(1)

    transformation_matrix = None
    for point_cloud_source_filename, point_cloud_target_filename in zip(
        point_cloud_source_file_names, point_cloud_target_file_names
    ):
        print("processing point cloud with index: ", str(idx_file))
        point_cloud_source, header_source = read_point_cloud_with_intensity(
            os.path.join(args.folder_path_point_cloud_source, point_cloud_source_filename)
        )
        point_cloud_target, header_target = read_point_cloud_with_intensity(
            os.path.join(args.folder_path_point_cloud_target, point_cloud_target_filename)
        )

        normalize_intensity(point_cloud_source)
        normalize_intensity(point_cloud_target)
        point_cloud_source = filter_point_cloud(point_cloud_source, 200)
        point_cloud_target = filter_point_cloud(point_cloud_target, 200)
        fitness_best = 0.0
        if idx_file >= 0:
            if transformation_matrix_best is None:
                transformation_matrix_best = transformation_matrix_initial
            transformation_matrix, inlier_rmse_best, fitness_best = register_point_clouds(
                do_initial_registration,
                point_cloud_source,
                point_cloud_target,
                idx_file,
                transformation_matrix_best,
                args.initial_voxel_size,
                args.continuous_voxel_size,
            )
            if inlier_rmse_best < inlier_rmse_global_best:
                if round(inlier_rmse_best, 2) == 0.00:
                    print("ERROR: inlier_rmse_best is 0.0")
                    exit(1)
                inlier_rmse_global_best = inlier_rmse_best
            if fitness_best > fitness_global_best:
                fitness_global_best = fitness_best

            loss = inlier_rmse_best + 1 / fitness_best
            if loss < loss_best:
                loss_best = loss
                transformation_matrix_best = transformation_matrix
        else:
            transformation_matrix = transformation_matrix_initial

        if transformation_matrix is not None:
            # 1. transform point cloud from robosense lidar to gps/rtk device coordinate system
            one_column = np.ones((len(point_cloud_source), 1), dtype=float)
            point_cloud_source_homogeneous = np.concatenate((point_cloud_source[:, 0:3], one_column), axis=1)
            source_transformed = np.matmul(transformation_matrix, point_cloud_source_homogeneous.T).T
            source_transformed[:, 3] = point_cloud_source[:, 3]
        else:
            source_transformed = point_cloud_source

        points_stacked = np.vstack([source_transformed, point_cloud_target])

        point_cloud_source_filename = point_cloud_target_filename.split(".")[0]
        seconds = int(point_cloud_target_filename.split("_")[0])
        nano_seconds = int(point_cloud_target_filename.split("_")[1])
        if bool(args.save_registered_point_clouds):
            store_fitness = False
            if store_fitness:
                file_name = (
                    str("{0:.3f}".format(fitness_best))
                    + "_fitness_"
                    + str(seconds)
                    + "_"
                    + str(nano_seconds).zfill(9)
                    + "_registered_point_clouds.pcd"
                )
            else:
                file_name = (
                    str(seconds)
                    + "_"
                    + str(nano_seconds).zfill(9)
                    + "_s110_lidar_ouster_south_and_north_registered.pcd"
                )
            write_point_cloud_with_intensity(
                args.output_folder_path_registered_point_clouds + "/" + file_name, points_stacked, header_source
            )
            # write fitness into log file:
            write_log_file = True
            if write_log_file:
                with open(args.output_folder_path_registered_point_clouds + "/registration_log.txt", "a") as log_file:
                    log_file.write(file_name + ": " + str(fitness_best) + "\n")
        idx_file = idx_file + 1

    print("==========================")
    print("Global results:")
    print("Best global RMSE: %.4f" % inlier_rmse_global_best)
    print("Best global fitness: %.4f" % fitness_global_best)
    print("Final loss: %.4f" % loss_best)
    print("Best global transformation_matrix: \n", repr(transformation_matrix))
    # store global best transformation matrix in log file
    with open(
        Path(args.output_folder_path_registered_point_clouds).parent.absolute().as_posix() + "/registration_log.txt",
        "a",
    ) as log_file:
        log_file.write("Best global RMSE: %.4f" % inlier_rmse_global_best + "\n")
        log_file.write("Best global fitness: %.4f" % fitness_global_best + "\n")
        log_file.write("Final loss: %.4f" % loss_best + "\n")
        log_file.write("Best global transformation_matrix: \n" + repr(transformation_matrix) + "\n")
    sys.exit()
