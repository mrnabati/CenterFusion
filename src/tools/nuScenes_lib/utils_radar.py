import copy
import cv2
import numpy as np
import os.path as osp
from functools import reduce
from pyquaternion import Quaternion
from typing import Tuple, List, Dict
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix


def map_pointcloud_to_image(pc, cam_intrinsic, img_shape=(1600,900)):
    """
    Map point cloud from camera coordinates to the image
    
    :param pc (PointCloud): point cloud in camera
    :param cam_cs_record (dict): Camera calibrated sensor record
    :param img_shape: shape of the image (width, height) 
    :return points (nparray), depth, mask: Mapped and filtered points with depth and mask
    """

    if isinstance(pc, RadarPointCloud):
        points = pc.points[:3,:]
    else:
        points = pc

    (width, height) = img_shape
    depths = points[2, :]
    
    ## Take the actual picture
    points = view_points(points[:3, :], cam_intrinsic, normalize=True)

    ## Remove points that are either outside or behind the camera. 
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < width - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < height - 1)
    points = points[:, mask]
    depths = depths[mask]

    return points, depths, mask


def get_radar_multisweep(nusc: 'NuScenes',
                         sample_rec: Dict,
                         chan: str,
                         ref_chan: str,
                         nsweeps: int = 5,
                         min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
    """
    Return a point cloud that aggregates multiple sweeps.
    As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
    As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
    :param nusc: A NuScenes instance.
    :param sample_rec: The current sample.
    :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
    :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
    :param nsweeps: Number of sweeps to aggregated.
    :param min_distance: Distance below which points are discarded.
    :return: (all_pc, all_times). The aggregated point cloud and timestamps.
    """
    # Init.
    all_pc = RadarPointCloud(np.zeros((18, 0)))
    all_times = np.zeros((1, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame.
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data'][chan]
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = RadarPointCloud.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
        times = time_lag * np.ones((1, current_pc.nbr_points()))
        all_times = np.hstack((all_times, times))

        # Merge with key pc.
        all_pc.points = np.hstack((all_pc.points, current_pc.points))

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return all_pc, all_times