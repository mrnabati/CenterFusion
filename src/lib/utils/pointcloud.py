from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import RadarPointCloud
from functools import reduce
from typing import Tuple, Dict
from model.utils import _topk, _tranpose_and_gather_feat
import os.path as osp
import torch
import timeit

import numpy as np
from pyquaternion import Quaternion

def map_pointcloud_to_image(pc, cam_intrinsic, img_shape=(1600,900)):
    """
    Map point cloud from camera coordinates to the image
    
    :param pc (PointCloud): point cloud in vehicle or global coordinates
    :param cam_cs_record (dict): Camera calibrated sensor record
    :param img_shape: shape of the image (width, height)
    :param coordinates (str): Point cloud coordinates ('vehicle', 'global') 
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
    points[2,:] = depths[mask]

    return points, mask


## A RadarPointCloud class where Radar velocity values are correctly 
# transformed to the target coordinate system
class RadarPointCloudWithVelocity(RadarPointCloud):
    
    @classmethod
    def rotate_velocity(cls, pointcloud, transform_matrix):
        n_points = pointcloud.shape[1]
        third_dim = np.zeros(n_points)
        pc_velocity = np.vstack((pointcloud[[8,9], :], third_dim, np.ones(n_points)))
        pc_velocity = transform_matrix.dot(pc_velocity)
        
        ## in camera coordinates, x is right, z is front
        pointcloud[[8,9],:] = pc_velocity[[0,2],:]

        return pointcloud


    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
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
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
        ref_from_car_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)
        car_from_global_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)
            global_from_car_rot = transform_matrix([0.0, 0.0, 0.0],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)
            car_from_current_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(current_cs_rec['rotation']), inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            velocity_trans_matrix = reduce(np.dot, [ref_from_car_rot, car_from_global_rot, global_from_car_rot, car_from_current_rot])
            current_pc.transform(trans_matrix)

            # Do the required rotations to the Radar velocity values
            current_pc.points = cls.rotate_velocity(current_pc.points, velocity_trans_matrix)

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


def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = torch.atan2(rot[:, 2], rot[:, 3]) + (-0.5 * 3.14159)
  alpha2 = torch.atan2(rot[:, 6], rot[:, 7]) + ( 0.5 * 3.14159)
  # return alpha1 * idx + alpha2 * (~idx)
  alpha = alpha1 * idx.float() + alpha2 * (~idx).float()
  return alpha


def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + torch.atan2(x - cx, fx)
    if rot_y > 3.14159:
      rot_y -= 2 * 3.14159
    if rot_y < -3.14159:
      rot_y += 2 * 3.14159
    return rot_y


def comput_corners_3d(dim, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = torch.cos(rotation_y), torch.sin(rotation_y)
    R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = torch.tensor([x_corners, y_corners, z_corners], dtype=torch.float32)
    corners_3d = torch.mm(R, corners).transpose(1, 0)
    return corners_3d


def get_dist_thresh(calib, ct, dim, alpha):
    rotation_y = alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0])
    corners_3d = comput_corners_3d(dim, rotation_y)
    dist_thresh = max(corners_3d[:,2]) - min(corners_3d[:,2]) / 2.0
    return dist_thresh


def generate_pc_hm(output, pc_dep, calib, opt):
      K = opt.K
      # K = 100
      heat = output['hm']
      wh = output['wh']
      pc_hm = torch.zeros_like(pc_dep)

      batch, cat, height, width = heat.size()
      scores, inds, clses, ys0, xs0 = _topk(heat, K=K)
      xs = xs0.view(batch, K, 1) + 0.5
      ys = ys0.view(batch, K, 1) + 0.5
      
      ## Initialize pc_feats
      pc_feats = torch.zeros((batch, len(opt.pc_feat_lvl), height, width), device=heat.device)
      dep_ind = opt.pc_feat_channels['pc_dep']
      vx_ind = opt.pc_feat_channels['pc_vx']
      vz_ind = opt.pc_feat_channels['pc_vz']
      to_log = opt.sigmoid_dep_sec
      
      ## get estimated depths
      out_dep = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      dep = _tranpose_and_gather_feat(out_dep, inds) # B x K x (C)
      if dep.size(2) == cat:
        cats = clses.view(batch, K, 1, 1)
        dep = dep.view(batch, K, -1, 1) # B x K x C x 1
        dep = dep.gather(2, cats.long()).squeeze(2) # B x K x 1

      ## get top bounding boxes
      wh = _tranpose_and_gather_feat(wh, inds) # B x K x 2
      wh = wh.view(batch, K, 2)
      wh[wh < 0] = 0
      if wh.size(2) == 2 * cat: # cat spec
        wh = wh.view(batch, K, -1, 2)
        cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
        wh = wh.gather(2, cats.long()).squeeze(2) # B x K x 2
      bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                          ys - wh[..., 1:2] / 2,
                          xs + wh[..., 0:1] / 2, 
                          ys + wh[..., 1:2] / 2], dim=2)  # B x K x 4
      
      ## get dimensions
      dims = _tranpose_and_gather_feat(output['dim'], inds).view(batch, K, -1)

      ## get rotation
      rot = _tranpose_and_gather_feat(output['rot'], inds).view(batch, K, -1)

      ## Calculate values for the new pc_hm
      clses = clses.cpu().numpy()

      for i, [pc_dep_b, bboxes_b, depth_b, dim_b, rot_b] in enumerate(zip(pc_dep, bboxes, dep, dims, rot)):
        alpha_b = get_alpha(rot_b).unsqueeze(1)

        if opt.sort_det_by_dist:
          idx = torch.argsort(depth_b[:,0])
          bboxes_b = bboxes_b[idx,:]
          depth_b = depth_b[idx,:]
          dim_b = dim_b[idx,:]
          rot_b = rot_b[idx,:]
          alpha_b = alpha_b[idx,:]

        for j, [bbox, depth, dim, alpha] in enumerate(zip(bboxes_b, depth_b, dim_b, alpha_b)):
          clss = clses[i,j].tolist()
          ct = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], device=pc_dep_b.device)
          dist_thresh = get_dist_thresh(calib, ct, dim, alpha)
          dist_thresh += dist_thresh * opt.frustumExpansionRatio
          pc_dep_to_hm_torch(pc_hm[i], pc_dep_b, depth, bbox, dist_thresh, opt)
      return pc_hm


def pc_dep_to_hm_torch(pc_hm, pc_dep, dep, bbox, dist_thresh, opt):
    if isinstance(dep, list) and len(dep) > 0:
      dep = dep[0]
    ct = torch.tensor(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=torch.float32)
    bbox_int = torch.tensor([torch.floor(bbox[0]), 
                         torch.floor(bbox[1]), 
                         torch.ceil(bbox[2]), 
                         torch.ceil(bbox[3])], dtype=torch.int32)# format: xyxy

    roi = pc_dep[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
    pc_dep = roi[opt.pc_feat_channels['pc_dep']]
    pc_vx = roi[opt.pc_feat_channels['pc_vx']]
    pc_vz = roi[opt.pc_feat_channels['pc_vz']]

    pc_dep.sum().data
    nonzero_inds = torch.nonzero(pc_dep, as_tuple=True)
    
    if len(nonzero_inds) and len(nonzero_inds[0]) > 0:
    #   nonzero_pc_dep = torch.exp(-pc_dep[nonzero_inds])
      nonzero_pc_dep = pc_dep[nonzero_inds]
      nonzero_pc_vx = pc_vx[nonzero_inds]
      nonzero_pc_vz = pc_vz[nonzero_inds]

      ## Get points within dist threshold
      within_thresh = (nonzero_pc_dep < dep+dist_thresh) \
              & (nonzero_pc_dep > max(0, dep-dist_thresh))
      pc_dep_match = nonzero_pc_dep[within_thresh]
      pc_vx_match = nonzero_pc_vx[within_thresh]
      pc_vz_match = nonzero_pc_vz[within_thresh]

      if len(pc_dep_match) > 0:
        arg_min = torch.argmin(pc_dep_match)
        dist = pc_dep_match[arg_min]
        vx = pc_vx_match[arg_min]
        vz = pc_vz_match[arg_min]
        if opt.normalize_depth:
          dist /= opt.max_pc_dist

        w = bbox[2] - bbox[0]
        w_interval = opt.hm_to_box_ratio*(w)
        w_min = int(ct[0] - w_interval/2.)
        w_max = int(ct[0] + w_interval/2.)
        
        h = bbox[3] - bbox[1]
        h_interval = opt.hm_to_box_ratio*(h)
        h_min = int(ct[1] - h_interval/2.)
        h_max = int(ct[1] + h_interval/2.)

        pc_hm[opt.pc_feat_channels['pc_dep'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = dist
        pc_hm[opt.pc_feat_channels['pc_vx'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = vx
        pc_hm[opt.pc_feat_channels['pc_vz'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = vz



def pc_dep_to_hm(pc_hm, pc_dep, dep, bbox, dist_thresh, opt):
    if isinstance(dep, list) and len(dep) > 0:
      dep = dep[0]
    ct = np.array(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    bbox_int = np.array([np.floor(bbox[0]), 
                         np.floor(bbox[1]), 
                         np.ceil(bbox[2]), 
                         np.ceil(bbox[3])], np.int32)# format: xyxy

    roi = pc_dep[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
    pc_dep = roi[opt.pc_feat_channels['pc_dep']]
    pc_vx = roi[opt.pc_feat_channels['pc_vx']]
    pc_vz = roi[opt.pc_feat_channels['pc_vz']]

    nonzero_inds = np.nonzero(pc_dep)
    
    if len(nonzero_inds[0]) > 0:
    #   nonzero_pc_dep = np.exp(-pc_dep[nonzero_inds])
      nonzero_pc_dep = pc_dep[nonzero_inds]
      nonzero_pc_vx = pc_vx[nonzero_inds]
      nonzero_pc_vz = pc_vz[nonzero_inds]

      ## Get points within dist threshold
      within_thresh = (nonzero_pc_dep < dep+dist_thresh) \
              & (nonzero_pc_dep > max(0, dep-dist_thresh))
      pc_dep_match = nonzero_pc_dep[within_thresh]
      pc_vx_match = nonzero_pc_vx[within_thresh]
      pc_vz_match = nonzero_pc_vz[within_thresh]

      if len(pc_dep_match) > 0:
        arg_min = np.argmin(pc_dep_match)
        dist = pc_dep_match[arg_min]
        vx = pc_vx_match[arg_min]
        vz = pc_vz_match[arg_min]
        if opt.normalize_depth:
          dist /= opt.max_pc_dist

        w = bbox[2] - bbox[0]
        w_interval = opt.hm_to_box_ratio*(w)
        w_min = int(ct[0] - w_interval/2.)
        w_max = int(ct[0] + w_interval/2.)
        
        h = bbox[3] - bbox[1]
        h_interval = opt.hm_to_box_ratio*(h)
        h_min = int(ct[1] - h_interval/2.)
        h_max = int(ct[1] + h_interval/2.)

        pc_hm[opt.pc_feat_channels['pc_dep'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = dist
        pc_hm[opt.pc_feat_channels['pc_vx'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = vx
        pc_hm[opt.pc_feat_channels['pc_vz'],
               h_min:h_max+1, 
               w_min:w_max+1+1] = vz
    