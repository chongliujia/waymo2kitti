import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from multiprocessing import Pool
from os.path import join, isdir
import argparse
from glob import glob

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

filter_empty_3dboxes = False
filter_no_label_zone_points = True
selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
selected_waymo_locations = None
save_track_id = True

class WaymoToKITTI(object):

    def __init__(self, load_dir, save_dir, prefix, num_proc):
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        self.type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.num_proc = int(num_proc)

        self.tfrecord_pathnames = sorted(glob(join(self.load_dir, '*.tfrecord')))

        self.label_save_dir = self.save_dir + '/label_'
        self.label_all_save_dir = self.save_dir + '/label_all'
        self.image_save_dir = self.save_dir + '/image_'
        self.calib_save_dir = self.save_dir + '/calib'
        self.point_cloud_save_dir = self.save_dir + '/velodyne'
        self.pose_save_dir = self.save_dir + '/pose'

        self.create_folder()

    def convert(self):
        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = list(tqdm.tqdm(p.imap(self.convert_one, range(len(self))), total=len(self)))
        print("\nfinished ...")

    def convert_one(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        for frame_idx, data in enumerate(dataset):

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if selected_waymo_locations is not None and frame.context.stats.location not in selected_waymo_locations:
                continue

            self.save_image(frame, file_idx, frame_idx)
            self.save_calib(frame, file_idx, frame_idx)
            self.save_lidar(frame, file_idx, frame_idx)
            self.save_label(frame, file_idx, frame_idx)
            self.save_pose(frame, file_idx, frame_idx)

    def __len__(self):
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        for img in frame.images:
            img_path = self.image_save_dir + str(img.name - 1) + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.png'
            img = cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.imsave(img_path, rgb_img, format='png')

    def save_calib(self, frame, file_idx, frame_idx):
        T_front_cam_to_ref = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ])

        for camera in frame.context.camera_calibrations:
            if camera.name == 1:
                T_front_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(4, 4)
                T_vehicle_to_front_cam = np.linalg.inv(T_front_cam_to_vehicle)

                front_cam_intrinsic = np.zeros((3, 4))
                front_cam_intrinsic[0, 0] = camera.intrinsic[0]
                front_cam_intrinsic[1, 1] = camera.intrinsic[1]
                front_cam_intrinsic[0, 2] = camera.intrinsic[2]
                front_cam_intrinsic[1, 2] = camera.intrinsic[3]
                front_cam_intrinsic[2, 2] = 1

                break

        self.T_front_cam_to_ref = T_front_cam_to_ref.copy()
        self.T_vehicle_to_front_cam = T_vehicle_to_front_cam.copy()

        identity_3x4 = np.eye(4)[:3, :]

        calib_context = ''
        for i in range(4):
            if i == 2:
                P2 = front_cam_intrinsic.reshape(12)
                calib_context += "P2: " + " ".join(['{}'.format(i) for i in P2]) + '\n'
            else:
                calib_context += "P" + str(i) + ": " + " ".join(['{}'.format(i) for i in identity_3x4.reshape(12)]) + '\n'

        calib_context += "R0_rect" + ": " + " ".join(['{}'.format(i) for i in np.eye(3).astype(np.float32).flatten()]) + '\n'

        Tr_velo_to_cam = self.cart_to_homo(T_front_cam_to_ref) @ np.linalg.inv(T_front_cam_to_vehicle)
        calib_context += "Tr_velo_to_cam" + ": " + " ".join(['{}'.format(i) for i in Tr_velo_to_cam[:3, :].reshape(12)]) + '\n'

        with open(self.calib_save_dir + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt', 'w+') as fp_calib:
            fp_calib.write(calib_context)

    def save_lidar(self, frame, file_idx, frame_idx):
        range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
        points_0, cp_points_0, intensity_0 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0
        )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)

        points_1, cp_points_1, intensity_1 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1
        )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)

        point_cloud = np.column_stack((points, intensity))
        pc_path = self.point_cloud_save_dir + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.bin'
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_label(self, frame, file_idx, frame_idx):
        fp_label_all = open(self.label_all_save_dir + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt', 'w+')
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            if bounding_box is None or name is None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in selected_waymo_classes:
                continue

            if filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            my_type = self.waymo_to_kitti_class_map[my_type]
            height = obj.box.height
            width = obj.box.width
            length = obj.box.length
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2

            pt_ref = self.cart_to_homo(self.T_front_cam_to_ref) @ self.T_vehicle_to_front_cam @ np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -obj.box.heading - np.pi / 2

            track_id = obj.id
            truncated = 0
            occluded = 0
            alpha = -10

            line = my_type + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(round(truncated, 2),
                                                                                   occluded,
                                                                                   round(alpha, 2),
                                                                                   round(bounding_box[0], 2),
                                                                                   round(bounding_box[1], 2),
                                                                                   round(bounding_box[2], 2),
                                                                                   round(bounding_box[3], 2),
                                                                                   round(height, 2),
                                                                                   round(width, 2),
                                                                                   round(length, 2),
                                                                                   round(x, 2),
                                                                                   round(y, 2),
                                                                                   round(z, 2),
                                                                                   round(rotation_y, 2))
            if save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            fp_label = open(self.label_save_dir + name + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt', 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def save_pose(self, frame, file_idx, frame_idx):
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(join(self.pose_save_dir, self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt'), pose)

    def create_folder(self):
        for d in [self.label_all_save_dir, self.calib_save_dir, self.point_cloud_save_dir, self.pose_save_dir]:
            if not isdir(d):
                os.makedirs(d)
        for d in [self.label_save_dir, self.image_save_dir]:
            for i in range(5):
                if not isdir(d + str(i)):
                    os.makedirs(d + str(i))

    def cart_to_homo(self, mat):
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--prefix', default='', help='Prefix to be added to converted file names')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = WaymoToKITTI(args.load_dir, args.save_dir, args.prefix, args.num_proc)
    converter.convert()
