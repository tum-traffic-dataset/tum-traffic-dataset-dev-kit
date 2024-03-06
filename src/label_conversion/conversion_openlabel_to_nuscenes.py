import argparse
import os
from os import path as osp
from glob import glob
from pypcd import pypcd  # solution to correctly installing this :https://github.com/dimatura/pypcd/issues/28
import mmcv
import numpy as np
import os.path
import shutil
import json


class OpenLABEL2NuScenesConverter(object):
    """
    OpenLABEL to nuScenes format converter.
    This class serves as the converter to change OpenLABEL datasets to nuScenes format.
    """

    def __init__(self, splits, load_dir, save_dir):
        """
        Args:
            splits list[(str)]: Contains the different splits
            load_dir (str): Directory to load openlabel data.
            save_dir (str): Directory to save data in nuScenes format.
        """

        self.splits = splits
        self.load_dir = load_dir
        self.save_dir = save_dir

        self.train_set = []
        self.val_set = []
        self.test_set = []

        self.map_set_to_dir_idx = {
            'training': 0,
            'validation': 1,
            'testing': 2
        }

        self.map_version_to_dir = {
            'training': 'train',
            'validation': 'val',
            'testing': 'test'
        }

        self.imagesets = {
            'training': self.train_set,
            'validation': self.val_set,
            'testing': self.test_set
        }

        self.occlusion_map = {
            'NOT_OCCLUDED': 0,
            'PARTIALLY_OCCLUDED': 1,
            'MOSTLY_OCCLUDED': 2
        }

        self.pickle = []

    def convert_openlabel_to_nuscenes(self):
        """Convert action."""
        print('Start converting ...')
        for split in self.splits:

            print(f'Converting split: {split}...')

            folder_path_point_clouds = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split],
                                                                'point_clouds', '*')))
            infos_list = []
            camera_labels_list = []
            for camera_list in sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_images',
                                                        '*'))):
                camera_labels_list.append(sorted(glob(os.path.join(camera_list, '*'))))

            for folder_path_point_cloud in folder_path_point_clouds:
                # get folder name
                folder_name = folder_path_point_cloud.split('/')[-1]
                self.create_folder(split, folder_name)
                pcd_list = sorted(glob(os.path.join(folder_path_point_cloud, '*')))
                point_cloud_save_dir = os.path.join(self.save_dir, self.map_version_to_dir[split], "point_clouds",
                                                    folder_name)
                self.convert_point_cloud(pcd_list, output_folder=point_cloud_save_dir)

                labels_list = sorted(
                    glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_point_clouds',
                                      folder_name, '*')))
                infos_list += self._fill_infos(labels_list, camera_labels_list)

            metadata = dict(version='r1')
            print("{} sample: {}".format(self.map_version_to_dir[split], len(infos_list)))
            data = dict(infos=infos_list, metadata=metadata)
            info_path = os.path.join(self.save_dir,
                                     "{}_infos_{}.pkl".format('nuscenes', self.map_version_to_dir[split]))
            mmcv.dump(data, info_path)
            # creating symbolic link for images to new dataset
            saving_loc = os.path.join(self.save_dir, self.map_version_to_dir[split], "images")
            if not os.path.isdir(saving_loc):
                os.symlink(os.path.join(self.load_dir, self.map_version_to_dir[split], "images"), saving_loc,
                           target_is_directory=True)

        print('\nFinished ...')

    def _fill_infos(self, labels_list, camera_labels_list):
        # using lidar as both the global and ego coordinate system
        # we use coordinates transformation from just 1 file because transformations are static in our case

        infos_list = []
        json_file = open(labels_list[0])
        openlable = json.loads(json_file.read())['openlabel']
        coordinate_systems = openlable['coordinate_systems']
        intrinsic = openlable['streams']

        # lidar_names = ['s110_lidar_ouster_south', 's110_lidar_ouster_north']
        lidar_names = coordinate_systems['s110_base']['children']
        # normally one lidar and 2 camera children
        cam_names = coordinate_systems[lidar_names[0]]['children']
        # cam_names = [
        #     's110_camera_basler_south1_8mm', 's110_camera_basler_south2_8mm'
        # ]

        lidar2ego = coordinate_systems[lidar_names[0]]['pose_wrt_parent']['matrix4x4']
        lidar2ego = np.array(lidar2ego).reshape((4, 4))
        lidar2ego = np.linalg.inv(lidar2ego)
        # lidar2ego = lidar2ego[:-1, :]

        south1intrinsics = intrinsic['s110_camera_basler_south1_8mm']['stream_properties']['intrinsics_pinhole'][
            'camera_matrix_3x4']
        south1intrinsics = np.array(south1intrinsics)[:, :-1]

        south12lidar = coordinate_systems['s110_camera_basler_south1_8mm']['pose_wrt_parent']['matrix4x4']
        south12lidar = np.array(south12lidar).reshape((4, 4))
        south12lidar = np.linalg.inv(south12lidar)
        # south12lidar = south12lidar[:-1, :]

        south12ego = lidar2ego @ south12lidar
        # south12ego = south12ego[:-1, :]

        south2intrinsics = intrinsic['s110_camera_basler_south1_8mm']['stream_properties']['intrinsics_pinhole'][
            'camera_matrix_3x4']
        south2intrinsics = np.array(south2intrinsics)[:, :-1]

        south22lidar = coordinate_systems['s110_camera_basler_south2_8mm']['pose_wrt_parent']['matrix4x4']
        south22lidar = np.array(south22lidar).reshape((4, 4))
        south22lidar = np.linalg.inv(south22lidar)
        # south22lidar = south22lidar[:-1, :]

        south22ego = lidar2ego @ south22lidar
        # south22ego = south22ego[:-1, :]

        ego2global = np.eye(4)

        for j, label_path in enumerate(labels_list):
            json1_file = open(label_path)
            json1_str = json1_file.read()
            annotations = json.loads(json1_str)

            cam_annotations = [next(iter(json.load(open(camera_labels_list[0][j]))['openlabel']['frames'].values())) \
                , next(iter(json.load(open(camera_labels_list[1][j]))['openlabel']['frames'].values()))]

            anno_frame = next(iter(annotations['openlabel']['frames'].values()))

            info = {
                'timestamp': anno_frame['frame_properties']['timestamp']
            }
            cam_infos = dict()
            for i, cam_name in enumerate(cam_names):
                cam_info = dict()
                cam_info['ego_pose'] = ego2global
                cam_info['height'] = 1200
                cam_info['width'] = 1920
                cam_info['filename'] = os.path.join(cam_annotations[i]['frame_properties']['image_file_name'])
                cam_info['calibrated_sensor'] = eval('south{}2ego'.format(i + 1))
                cam_info['camera_intrinsic'] = eval('south{}intrinsics'.format(i + 1))

                cam_infos[cam_name] = cam_info
            lidar_infos = dict()
            for lidar_name in lidar_names:
                lidar_info = dict()
                lidar_info['ego_pose'] = ego2global
                lidar_info['filename'] = os.path.join(
                    anno_frame['frame_properties']['point_cloud_file_name'].split('.')[0] + '.bin')
                lidar_info['calibrated_sensor'] = lidar2ego
                lidar_infos[lidar_name] = lidar_info

            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            ann_infos = list()

            for ann in anno_frame['objects']:
                ann_info = dict()
                ann_info['val'] = dict()
                val = anno_frame['objects'][ann]['object_data']['cuboid']['val']
                ann_info['val']['size'] = [val[8], val[7], val[9]]  # going from lwh -> wlh
                ann_info['val']['loc'] = val[:3]
                ann_info['val']['rot'] = np.roll(val[3:7], 1).tolist()  # going from xyzw -> wxyz
                ann_info['type'] = anno_frame['objects'][ann]['object_data']['type']
                ann_infos.append(ann_info)
            info['ann_infos'] = ann_infos
            infos_list.append(info)

        return infos_list

    @staticmethod
    def save_lidar(file, out_file):
        """
        Converts file from .pcd to .bin
        Args:
            file: Filepath to .pcd
            out_file: Filepath of .bin
        """
        point_cloud = pypcd.PointCloud.from_path(file)
        np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32) / 256
        np_ts = np.zeros((np_x.shape[0],), dtype=np.float32)
        bin_format = np.column_stack((np_x, np_y, np_z, np_i, np_ts)).flatten()
        bin_format.tofile(os.path.join(f'{out_file}.bin'))

    @staticmethod
    def save_img(file, out_file):
        """
        Copies images to new location
        Args:
            file: Path to image
            out_file: Path to new location
        """
        img_path = f'{out_file}.jpg'
        shutil.copyfile(file, img_path)

    def create_folder(self, split, folder_name):
        """
        Create folder for data preprocessing.
        """

        # dir_list1 = [f'point_clouds/s110_lidar_ouster_south', f'point_clouds/s110_lidar_ouster_north']
        # for d in dir_list1:
        point_cloud_save_dir = os.path.join(self.save_dir, self.map_version_to_dir[split], f'point_clouds', folder_name)
        os.makedirs(point_cloud_save_dir, exist_ok=True, mode=0o777)

    def convert_point_cloud(self, pcd_list, output_folder):
        """
        Convert point cloud from pcd format to bin
        """
        for idx, pcd in enumerate(pcd_list):
            out_filename = pcd.split('/')
            out_filename = out_filename[-1]
            out_filename = out_filename[:-4]
            self.save_lidar(pcd, os.path.join(output_folder, out_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-path',
        type=str,
        default='/data/00_tum_traffic_dataset/r02_tum_traffic_intersection_dataset/r02_tum_traffic_intersection_dataset_train_val_test/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='/data/00_tum_traffic_dataset/r02_tum_traffic_intersection_dataset/r02_tum_traffic_intersection_dataset_train_val_test_nuscenes_format/',
        required=False,
        help='name of info pkl')
    args = parser.parse_args()

    # splits = ['training', 'validation', 'testing']
    splits = ['training', 'validation']
    load_dir = osp.join(args.root_path)
    save_dir = osp.join(args.out_dir)
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    converter = OpenLABEL2NuScenesConverter(splits, load_dir, save_dir)
    converter.convert_openlabel_to_nuscenes()
