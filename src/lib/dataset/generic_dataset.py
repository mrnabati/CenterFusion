from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json
import cv2
import os
from collections import defaultdict
import time

import pycocotools.coco as coco
import torch
import torch.utils.data as data

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, gaussian2D
from utils.pointcloud import map_pointcloud_to_image, pc_dep_to_hm
import copy
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from utils.ddd_utils import compute_box_3d, project_to_image, draw_box_3d
from utils.ddd_utils import comput_corners_3d, alpha2rot_y, get_pc_hm


def get_dist_thresh(calib, ct, dim, alpha):
    rotation_y = alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0])
    corners_3d = comput_corners_3d(dim, rotation_y)
    dist_thresh = max(corners_3d[:,2]) - min(corners_3d[:,2]) / 2.0
    return dist_thresh


class GenericDataset(data.Dataset):
  default_resolution = None
  num_categories = None
  class_name = None
  # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
  # Not using 0 because 0 is used for don't care region and ignore loss.
  cat_ids = None
  max_objs = None
  rest_focal_length = 1200
  num_joints = 17
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
           [4, 6], [3, 5], [5, 6], 
           [5, 7], [7, 9], [6, 8], [8, 10], 
           [6, 12], [5, 11], [11, 12], 
           [12, 14], [14, 16], [11, 13], [13, 15]]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                      dtype=np.float32)
  _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
  ignore_val = 1
  nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4], 
    4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7], 7: [5, 6, 7]}
  
  ## change these vectors to actual mean and std to normalize
  pc_mean = np.zeros((18,1))
  pc_std = np.ones((18,1))
  img_ind = 0


  def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
    super(GenericDataset, self).__init__()
    if opt is not None and split is not None:
      self.split = split
      self.opt = opt
      self._data_rng = np.random.RandomState(123)
      self.enable_meta = True if (opt.run_dataset_eval and split in ["val", "mini_val", "test"]) or opt.eval else False
    
    if ann_path is not None and img_dir is not None:
      print('==> initializing {} data from {}, \n images from {} ...'.format(
        split, ann_path, img_dir))
      self.coco = coco.COCO(ann_path)
      self.images = self.coco.getImgIds()

      if opt.tracking:
        if not ('videos' in self.coco.dataset):
          self.fake_video_data()
        print('Creating video index!')
        self.video_to_images = defaultdict(list)
        for image in self.coco.dataset['images']:
          self.video_to_images[image['video_id']].append(image)
      
      self.img_dir = img_dir


  def __getitem__(self, index):
    opt = self.opt
    img, anns, img_info, img_path = self._load_data(index)
    height, width = img.shape[0], img.shape[1]

    ## sort annotations based on depth form far to near
    new_anns = sorted(anns, key=lambda k: k['depth'], reverse=True)

    ## Get center and scale from image
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
      else np.array([img.shape[1], img.shape[0]], np.float32)
    aug_s, rot, flipped = 1, 0, 0

    ## data augmentation for training set
    if 'train' in self.split:
      c, aug_s, rot = self._get_aug_param(c, s, width, height)
      s = s * aug_s
      if np.random.random() < opt.flip:
        flipped = 1
        img = img[:, ::-1, :]
        anns = self._flip_anns(anns, width)

    trans_input = get_affine_transform(
      c, s, rot, [opt.input_w, opt.input_h])
    trans_output = get_affine_transform(
      c, s, rot, [opt.output_w, opt.output_h])
    inp = self._get_input(img, trans_input)
    ret = {'image': inp}
    gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

    #  load point cloud data
    if opt.pointcloud:
      pc_2d, pc_N, pc_dep, pc_3d = self._load_pc_data(img, img_info, 
        trans_input, trans_output, flipped)
      ret.update({ 'pc_2d': pc_2d,
                   'pc_3d': pc_3d,
                   'pc_N': pc_N,
                   'pc_dep': pc_dep })

    pre_cts, track_ids = None, None
    if opt.tracking:
      pre_image, pre_anns, frame_dist, pre_img_info = self._load_pre_data(
        img_info['video_id'], img_info['frame_id'], 
        img_info['sensor_id'] if 'sensor_id' in img_info else 1)
      if flipped:
        pre_image = pre_image[:, ::-1, :].copy()
        pre_anns = self._flip_anns(pre_anns, width)
        if pc_2d is not None:
          pc_2d = self._flip_pc(pc_2d,  width)
      if opt.same_aug_pre and frame_dist != 0:
        trans_input_pre = trans_input 
        trans_output_pre = trans_output
      else:
        c_pre, aug_s_pre, _ = self._get_aug_param(
          c, s, width, height, disturb=True)
        s_pre = s * aug_s_pre
        trans_input_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.input_w, opt.input_h])
        trans_output_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.output_w, opt.output_h])
      pre_img = self._get_input(pre_image, trans_input_pre)
      pre_hm, pre_cts, track_ids = self._get_pre_dets(
        pre_anns, trans_input_pre, trans_output_pre)
      ret['pre_img'] = pre_img
      if opt.pre_hm:
        ret['pre_hm'] = pre_hm
      if opt.pointcloud:
        pre_pc_2d, pre_pc_N, pre_pc_hm, pre_pc_3d = self._load_pc_data(pre_img, pre_img_info, 
            trans_input_pre, trans_output_pre, flipped)
        ret['pre_pc_2d'] = pre_pc_2d
        ret['pre_pc_3d'] = pre_pc_3d
        ret['pre_pc_N'] = pre_pc_N
        ret['pre_pc_hm'] = pre_pc_hm

    ### init samples
    self._init_ret(ret, gt_det)
    calib = self._get_calib(img_info, width, height)

    # get velocity transformation matrix
    if "velocity_trans_matrix" in img_info:
      velocity_mat = np.array(img_info['velocity_trans_matrix'], dtype=np.float32)
    else:
      velocity_mat = np.eye(4)
    
    num_objs = min(len(anns), self.max_objs)
    for k in range(num_objs):
      ann = anns[k]
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -999:
        continue
      bbox, bbox_amodal = self._get_bbox_output(
        ann['bbox'], trans_output, height, width)
      if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
        self._mask_ignore_or_crowd(ret, cls_id, bbox)
        continue
      self._add_instance(
        ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
        calib, pre_cts, track_ids)

    if self.opt.debug > 0 or self.enable_meta:
      gt_det = self._format_gt_det(gt_det)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
              'img_path': img_path, 'calib': calib,
              'img_width': img_info['width'], 'img_height': img_info['height'],
              'flipped': flipped, 'velocity_mat':velocity_mat}
      ret['meta'] = meta
    ret['calib'] = calib
    return ret


  def get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib

  def _load_image_anns(self, img_id, coco, img_dir):
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

  def _load_data(self, index):
    coco = self.coco
    img_dir = self.img_dir
    img_id = self.images[index]
    img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

    return img, anns, img_info, img_path


  def _load_pre_data(self, video_id, frame_id, sensor_id=1):
    img_infos = self.video_to_images[video_id]
    # If training, random sample nearby frames as the "previous" frame
    # If testing, get the exact prevous frame
    if 'train' in self.split:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
          if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
          (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    else:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == -1 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
      if len(img_ids) == 0:
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == 0 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    rand_id = np.random.choice(len(img_ids))
    img_id, pre_frame_id = img_ids[rand_id]
    frame_dist = abs(frame_id - pre_frame_id)
    img, anns, img_info, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
    return img, anns, frame_dist, img_info


  def _get_pre_dets(self, anns, trans_input, trans_output):
    hm_h, hm_w = self.opt.input_h, self.opt.input_w
    down_ratio = self.opt.down_ratio
    trans = trans_input
    reutrn_hm = self.opt.pre_hm
    pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
    pre_cts, track_ids = [], []
    for ann in anns:
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -99 or \
         ('iscrowd' in ann and ann['iscrowd'] > 0):
        continue
      bbox = self._coco_box_to_bbox(ann['bbox'])
      bbox[:2] = affine_transform(bbox[:2], trans)
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      max_rad = 1
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius)) 
        max_rad = max(max_rad, radius)
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct0 = ct.copy()
        conf = 1

        ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
        ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
        conf = 1 if np.random.random() > self.opt.lost_disturb else 0
        
        ct_int = ct.astype(np.int32)
        if conf == 0:
          pre_cts.append(ct / down_ratio)
        else:
          pre_cts.append(ct0 / down_ratio)

        track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
        if reutrn_hm:
          draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

        if np.random.random() < self.opt.fp_disturb and reutrn_hm:
          ct2 = ct0.copy()
          # Hard code heatmap disturb ratio, haven't tried other numbers.
          ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
          ct2[1] = ct2[1] + np.random.randn() * 0.05 * h 
          ct2_int = ct2.astype(np.int32)
          draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

    return pre_hm, pre_cts, track_ids

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


  def _get_aug_param(self, c, s, width, height, disturb=False):
    if (not self.opt.not_rand_crop) and not disturb:
      aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
      w_border = self._get_border(128, width)
      h_border = self._get_border(128, height)
      c[0] = np.random.randint(low=w_border, high=width - w_border)
      c[1] = np.random.randint(low=h_border, high=height - h_border)
    else:
      sf = self.opt.scale
      cf = self.opt.shift
      # if type(s) == float:
      #   s = [s, s]
      temp = np.random.randn()*cf
      c[0] += s * np.clip(temp, -2*cf, 2*cf)
      c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      aug_s = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    
    if np.random.random() < self.opt.aug_rot:
      rf = self.opt.rotate
      rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)
    else:
      rot = 0
    
    return c, aug_s, rot


  def _flip_anns(self, anns, width):
    for k in range(len(anns)):
      bbox = anns[k]['bbox']
      anns[k]['bbox'] = [
        width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]
      
      if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
        keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(
          self.num_joints, 3)
        keypoints[:, 0] = width - keypoints[:, 0] - 1
        for e in self.flip_idx:
          keypoints[e[0]], keypoints[e[1]] = \
            keypoints[e[1]].copy(), keypoints[e[0]].copy()
        anns[k]['keypoints'] = keypoints.reshape(-1).tolist()

      if 'rot' in self.opt.heads and 'alpha' in anns[k]:
        anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 \
                           else - np.pi - anns[k]['alpha']

      if 'amodel_offset' in self.opt.heads and 'amodel_center' in anns[k]:
        anns[k]['amodel_center'][0] = width - anns[k]['amodel_center'][0] - 1

      if self.opt.velocity and 'velocity' in anns[k]:
        # anns[k]['velocity'] = [-10000, -10000, -10000]
        anns[k]['velocity'][0] *= -1

    return anns



  ## Load the Radar point cloud data
  def _load_pc_data(self, img, img_info, inp_trans, out_trans, flipped=0):
    img_height, img_width = img.shape[0], img.shape[1]
    radar_pc = np.array(img_info.get('radar_pc', None))
    if radar_pc is None:
      return None, None, None, None

    # calculate distance to points
    depth = radar_pc[2,:]
    
    # filter points by distance
    if self.opt.max_pc_dist > 0:
      mask = (depth <= self.opt.max_pc_dist)
      radar_pc = radar_pc[:,mask]
      depth = depth[mask]

    # add z offset to radar points
    if self.opt.pc_z_offset != 0:
      radar_pc[1,:] -= self.opt.pc_z_offset
    
    # map points to the image and filter ones outside
    pc_2d, mask = map_pointcloud_to_image(radar_pc, np.array(img_info['camera_intrinsic']), 
                              img_shape=(img_info['width'],img_info['height']))
    pc_3d = radar_pc[:,mask]

    # sort points by distance
    ind = np.argsort(pc_2d[2,:])
    pc_2d = pc_2d[:,ind]
    pc_3d = pc_3d[:,ind]

    # flip points if image is flipped
    if flipped:
      pc_2d = self._flip_pc(pc_2d,  img_width)
      pc_3d[0,:] *= -1  # flipping the x dimension
      pc_3d[8,:] *= -1  # flipping x velocity (x is right, z is front)

    pc_2d, pc_3d, pc_dep = self._process_pc(pc_2d, pc_3d, img, inp_trans, out_trans, img_info)
    pc_N = np.array(pc_2d.shape[1])

    # pad point clouds with zero to avoid size mismatch error in dataloader
    n_points = min(self.opt.max_pc, pc_2d.shape[1])
    pc_z = np.zeros((pc_2d.shape[0], self.opt.max_pc))
    pc_z[:, :n_points] = pc_2d[:, :n_points]
    pc_3dz = np.zeros((pc_3d.shape[0], self.opt.max_pc))
    pc_3dz[:, :n_points] = pc_3d[:, :n_points]

    return pc_z, pc_N, pc_dep, pc_3dz



  def _process_pc(self, pc_2d, pc_3d, img, inp_trans, out_trans, img_info):    
    img_height, img_width = img.shape[0], img.shape[1]

    # transform points
    mask = None
    if len(self.opt.pc_feat_lvl) > 0:
      pc_feat, mask = self._transform_pc(pc_2d, out_trans, self.opt.output_w, self.opt.output_h)
      pc_hm_feat = np.zeros((len(self.opt.pc_feat_lvl), self.opt.output_h, self.opt.output_w), np.float32)
    
    if mask is not None:
      pc_N = np.array(sum(mask))
      pc_2d = pc_2d[:,mask]
      pc_3d = pc_3d[:,mask]
    else:
      pc_N = pc_2d.shape[1]

    # create point cloud pillars
    if self.opt.pc_roi_method == "pillars":
      pillar_wh = self.create_pc_pillars(img, img_info, pc_2d, pc_3d, inp_trans, out_trans)    

    # generate point cloud channels
    for i in range(pc_N-1, -1, -1):
      for feat in self.opt.pc_feat_lvl:
        point = pc_feat[:,i]
        depth = point[2]
        ct = np.array([point[0], point[1]])
        ct_int = ct.astype(np.int32)

        if self.opt.pc_roi_method == "pillars":
          wh = pillar_wh[:,i]
          b = [max(ct[1]-wh[1], 0), 
              ct[1], 
              max(ct[0]-wh[0]/2, 0), 
              min(ct[0]+wh[0]/2, self.opt.output_w)]
          b = np.round(b).astype(np.int32)
        
        elif self.opt.pc_roi_method == "hm":
          radius = (1.0 / depth) * self.opt.r_a + self.opt.r_b
          radius = gaussian_radius((radius, radius))
          radius = max(0, int(radius))
          x, y = ct_int[0], ct_int[1]
          height, width = pc_hm_feat.shape[1:3]
          left, right = min(x, radius), min(width - x, radius + 1)
          top, bottom = min(y, radius), min(height - y, radius + 1)
          b = np.array([y - top, y + bottom, x - left, x + right])
          b = np.round(b).astype(np.int32)
        
        if feat == 'pc_dep':
          channel = self.opt.pc_feat_channels['pc_dep']
          pc_hm_feat[channel, b[0]:b[1], b[2]:b[3]] = depth
        
        if feat == 'pc_vx':
          vx = pc_3d[8,i]
          channel = self.opt.pc_feat_channels['pc_vx']
          pc_hm_feat[channel, b[0]:b[1], b[2]:b[3]] = vx
        
        if feat == 'pc_vz':
          vz = pc_3d[9,i]
          channel = self.opt.pc_feat_channels['pc_vz']
          pc_hm_feat[channel, b[0]:b[1], b[2]:b[3]] = vz

    return pc_2d, pc_3d, pc_hm_feat


  def create_pc_pillars(self, img, img_info, pc_2d, pc_3d, inp_trans, out_trans):
    pillar_wh = np.zeros((2, pc_3d.shape[1]))
    boxes_2d = np.zeros((0,8,2))
    pillar_dim = self.opt.pillar_dims
    v = np.dot(np.eye(3), np.array([1,0,0]))
    ry = -np.arctan2(v[2], v[0])

    for i, center in enumerate(pc_3d[:3,:].T):
      # Create a 3D pillar at pc location for the full-size image
      box_3d = compute_box_3d(dim=pillar_dim, location=center, rotation_y=ry)
      box_2d = project_to_image(box_3d, img_info['calib']).T  # [2x8]        
      
      ## save the box for debug plots
      if self.opt.debug:
        box_2d_img, m = self._transform_pc(box_2d, inp_trans, self.opt.input_w, 
                                            self.opt.input_h, filter_out=False)
        boxes_2d = np.concatenate((boxes_2d, np.expand_dims(box_2d_img.T,0)),0)

      # transform points
      box_2d_t, m = self._transform_pc(box_2d, out_trans, self.opt.output_w, self.opt.output_h)
      
      if box_2d_t.shape[1] <= 1:
        continue

      # get the bounding box in [xyxy] format
      bbox = [np.min(box_2d_t[0,:]), 
              np.min(box_2d_t[1,:]), 
              np.max(box_2d_t[0,:]), 
              np.max(box_2d_t[1,:])] # format: xyxy

      # store height and width of the 2D box
      pillar_wh[0,i] = bbox[2] - bbox[0]
      pillar_wh[1,i] = bbox[3] - bbox[1]

    ## DEBUG #################################################################
    if self.opt.debug:
      img_2d = copy.deepcopy(img)
      # img_3d = copy.deepcopy(img)
      img_2d_inp = cv2.warpAffine(img, inp_trans, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
      img_2d_out = cv2.warpAffine(img, out_trans, 
                        (self.opt.output_w, self.opt.output_h),
                        flags=cv2.INTER_LINEAR)
      img_3d = cv2.warpAffine(img, inp_trans, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
      blank_image = 255*np.ones((self.opt.input_h,self.opt.input_w,3), np.uint8)
      overlay = img_2d_inp.copy()
      output = img_2d_inp.copy()

      pc_inp, _= self._transform_pc(pc_2d, inp_trans, self.opt.input_w, self.opt.input_h)
      pc_out, _= self._transform_pc(pc_2d, out_trans, self.opt.output_w, self.opt.output_h)

      pill_wh_inp = pillar_wh * (self.opt.input_w/self.opt.output_w)
      pill_wh_out = pillar_wh
      pill_wh_ori = pill_wh_inp * 2
      
      for i, p in enumerate(pc_inp[:3,:].T):
        color = int((p[2].tolist()/60.0)*255)
        color = (0,color,0)
        
        rect_tl = (np.min(int(p[0]-pill_wh_inp[0,i]/2), 0), np.min(int(p[1]-pill_wh_inp[1,i]),0))
        rect_br = (np.min(int(p[0]+pill_wh_inp[0,i]/2), 0), int(p[1]))
        cv2.rectangle(img_2d_inp, rect_tl, rect_br, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        img_2d_inp = cv2.circle(img_2d_inp, (int(p[0]), int(p[1])), 3, color, -1)

        ## On original-sized image
        rect_tl_ori = (np.min(int(pc_2d[0,i]-pill_wh_ori[0,i]/2), 0), np.min(int(pc_2d[1,i]-pill_wh_ori[1,i]),0))
        rect_br_ori = (np.min(int(pc_2d[0,i]+pill_wh_ori[0,i]/2), 0), int(pc_2d[1,i]))
        cv2.rectangle(img_2d, rect_tl_ori, rect_br_ori, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        img_2d = cv2.circle(img_2d, (int(pc_2d[0,i]), int(pc_2d[1,i])), 6, color, -1)
        
        p2 = pc_out[:3,i].T
        rect_tl2 = (np.min(int(p2[0]-pill_wh_out[0,i]/2), 0), np.min(int(p2[1]-pill_wh_out[1,i]),0))
        rect_br2 = (np.min(int(p2[0]+pill_wh_out[0,i]/2), 0), int(p2[1]))
        cv2.rectangle(img_2d_out, rect_tl2, rect_br2, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        img_2d_out = cv2.circle(img_2d_out, (int(p[0]), int(p[1])), 3, (255,0,0), -1)
        
        # on blank image
        cv2.rectangle(blank_image, rect_tl, rect_br, color, -1, lineType=cv2.LINE_AA)
        
        # overlay
        alpha = 0.1
        cv2.rectangle(overlay, rect_tl, rect_br, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # plot 3d pillars
        img_3d = draw_box_3d(img_3d, boxes_2d[i].astype(np.int32), [114, 159, 207], 
                    same_color=False)

      cv2.imwrite((self.opt.debug_dir+ '/{}pc_pillar_2d_inp.' + self.opt.img_format)\
        .format(self.img_ind), img_2d_inp)
      cv2.imwrite((self.opt.debug_dir+ '/{}pc_pillar_2d_ori.' + self.opt.img_format)\
        .format(self.img_ind), img_2d)
      cv2.imwrite((self.opt.debug_dir+ '/{}pc_pillar_2d_out.' + self.opt.img_format)\
        .format(self.img_ind), img_2d_out)
      cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_2d_blank.'+ self.opt.img_format)\
        .format(self.img_ind), blank_image)
      cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_2d_overlay.'+ self.opt.img_format)\
        .format(self.img_ind), output)
      cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_3d.'+ self.opt.img_format)\
        .format(self.img_ind), img_3d)
      self.img_ind += 1
    ## DEBUG #################################################################
    return pillar_wh


  def _flip_pc(self, pc_2d, width):
    pc_2d[0,:] = width - 1 - pc_2d[0,:]
    return pc_2d
  

    # Transform points to image or feature space with augmentation
    #  Inputs:
    # pc_2d: [3xN]
  def _transform_pc(self, pc_2d, trans, img_width, img_height, filter_out=True):

    if pc_2d.shape[1] == 0:
      return pc_2d, []

    pc_t = np.expand_dims(pc_2d[:2,:].T, 0)   # [3,N] -> [1,N,2]
    t_points = cv2.transform(pc_t, trans)
    t_points = np.squeeze(t_points,0).T       # [1,N,2] -> [2,N]
    
    # remove points outside image
    if filter_out:
      mask = (t_points[0,:]<img_width) \
              & (t_points[1,:]<img_height) \
              & (0<t_points[0,:]) \
              & (0<t_points[1,:])
      out = np.concatenate((t_points[:,mask], pc_2d[2:,mask]), axis=0)
    else:
      mask = None
      out = np.concatenate((t_points, pc_2d[2:,:]), axis=0)

    return out, mask


  ## Augment, resize and normalize the image
  def _get_input(self, img, trans_input):
    inp = cv2.warpAffine(img, trans_input, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
    
    inp = (inp.astype(np.float32) / 255.)
    if 'train' in self.split and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    return inp


  def _init_ret(self, ret, gt_det):
    max_objs = self.max_objs * self.opt.dense_reg
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), 
      np.float32)
    ret['ind'] = np.zeros((max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((max_objs), dtype=np.float32)
    
    if self.opt.pointcloud:
      ret['pc_hm'] = np.zeros(
        (len(self.opt.pc_feat_lvl), self.opt.output_h, self.opt.output_w), 
        np.float32)

    regression_head_dims = {
      'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4, 
      'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2, 
      'dep': 1, 'dim': 3, 'amodel_offset': 2 }

    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head + '_mask'] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        gt_det[head] = []
    # if self.opt.pointcloud:
    #     ret['pc_dep_mask'] = np.zeros((max_objs, 1), dtype=np.float32)
    #     ret['pc_dep'] = np.zeros((max_objs, 1), dtype=np.float32)
    #     gt_det['pc_dep'] = []

    if 'hm_hp' in self.opt.heads:
      num_joints = self.num_joints
      ret['hm_hp'] = np.zeros(
        (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
      ret['hm_hp_mask'] = np.zeros(
        (max_objs * num_joints), dtype=np.float32)
      ret['hp_offset'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
      ret['hp_offset_mask'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)
    
    if 'rot' in self.opt.heads:
      ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
      ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
      ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
      gt_det.update({'rot': []})


  def _get_calib(self, img_info, width, height):
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _ignore_region(self, region, ignore_val=1):
    np.maximum(region, ignore_val, out=region)


  def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
    # mask out crowd region, only rectangular mask is supported
    if cls_id == 0: # ignore all classes
      self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                        int(bbox[0]): int(bbox[2]) + 1])
    else:
      # mask out one specific class
      self._ignore_region(ret['hm'][abs(cls_id) - 1, 
                                    int(bbox[1]): int(bbox[3]) + 1, 
                                    int(bbox[0]): int(bbox[2]) + 1])
    if ('hm_hp' in ret) and cls_id <= 1:
      self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                          int(bbox[0]): int(bbox[2]) + 1])


  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox


  def _get_bbox_output(self, bbox, trans_output, height, width):
    bbox = self._coco_box_to_bbox(bbox).copy()

    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
    for t in range(4):
      rect[t] =  affine_transform(rect[t], trans_output)
    bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
    bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

    bbox_amodal = copy.deepcopy(bbox)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    return bbox, bbox_amodal


  def _add_instance(
    self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
    aug_s, calib, pre_cts=None, track_ids=None):
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if h <= 0 or w <= 0:
      return
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius)) 
    ct = np.array(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    ret['cat'][k] = cls_id - 1
    ret['mask'][k] = 1
    if 'wh' in ret:
      ret['wh'][k] = 1. * w, 1. * h
      ret['wh_mask'][k] = 1
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
    ret['reg'][k] = ct - ct_int
    ret['reg_mask'][k] = 1
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

    gt_det['bboxes'].append(
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
    gt_det['scores'].append(1)
    gt_det['clses'].append(cls_id - 1)
    gt_det['cts'].append(ct)

    if 'tracking' in self.opt.heads:
      if ann['track_id'] in track_ids:
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        ret['tracking_mask'][k] = 1
        ret['tracking'][k] = pre_ct - ct_int
        gt_det['tracking'].append(ret['tracking'][k])
      else:
        gt_det['tracking'].append(np.zeros(2, np.float32))

    if 'ltrb' in self.opt.heads:
      ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
        bbox[2] - ct_int[0], bbox[3] - ct_int[1]
      ret['ltrb_mask'][k] = 1

    ## ltrb_amodal is to use the left, top, right, bottom bounding box representation 
    # to enable detecting out-of-image bounding box (important for MOT datasets)
    if 'ltrb_amodal' in self.opt.heads:
      ret['ltrb_amodal'][k] = \
        bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
        bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
      ret['ltrb_amodal_mask'][k] = 1
      gt_det['ltrb_amodal'].append(bbox_amodal)

    if 'nuscenes_att' in self.opt.heads:
      if ('attributes' in ann) and ann['attributes'] > 0:
        att = int(ann['attributes'] - 1)
        ret['nuscenes_att'][k][att] = 1
        ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
      gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

    if 'velocity' in self.opt.heads:
      if ('velocity_cam' in ann) and min(ann['velocity_cam']) > -1000:
        ret['velocity'][k] = np.array(ann['velocity_cam'], np.float32)[:3]
        ret['velocity_mask'][k] = 1
      gt_det['velocity'].append(ret['velocity'][k])

    if 'hps' in self.opt.heads:
      self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

    if 'rot' in self.opt.heads:
      self._add_rot(ret, ann, k, gt_det)

    if 'dep' in self.opt.heads:
      if 'depth' in ann:
        ret['dep_mask'][k] = 1
        ret['dep'][k] = ann['depth'] * aug_s
        gt_det['dep'].append(ret['dep'][k])
      else:
        gt_det['dep'].append(2)

    if 'dim' in self.opt.heads:
      if 'dim' in ann:
        ret['dim_mask'][k] = 1
        ret['dim'][k] = ann['dim']
        gt_det['dim'].append(ret['dim'][k])
      else:
        gt_det['dim'].append([1,1,1])
    
    if 'amodel_offset' in self.opt.heads:
      if 'amodel_center' in ann:
        amodel_center = affine_transform(ann['amodel_center'], trans_output)
        ret['amodel_offset_mask'][k] = 1
        ret['amodel_offset'][k] = amodel_center - ct_int
        gt_det['amodel_offset'].append(ret['amodel_offset'][k])
      else:
        gt_det['amodel_offset'].append([0, 0])
    
    if self.opt.pointcloud:
      ## get pointcloud heatmap
      if self.opt.disable_frustum:
        ret['pc_hm'] = ret['pc_dep']
        if opt.normalize_depth:
          ret['pc_hm'][self.opt.pc_feat_channels['pc_dep']] /= opt.max_pc_dist
      else:
        dist_thresh = get_dist_thresh(calib, ct, ann['dim'], ann['alpha'])
        pc_dep_to_hm(ret['pc_hm'], ret['pc_dep'], ann['depth'], bbox, dist_thresh, self.opt)
    
    

  def _add_hps(self, ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w):
    num_joints = self.num_joints
    pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3) \
        if 'keypoints' in ann else np.zeros((self.num_joints, 3), np.float32)
    if self.opt.simple_radius > 0:
      hp_radius = int(simple_radius(h, w, min_overlap=self.opt.simple_radius))
    else:
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))

    for j in range(num_joints):
      pts[j, :2] = affine_transform(pts[j, :2], trans_output)
      if pts[j, 2] > 0:
        if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_w and \
          pts[j, 1] >= 0 and pts[j, 1] < self.opt.output_h:
          ret['hps'][k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
          ret['hps_mask'][k, j * 2: j * 2 + 2] = 1
          pt_int = pts[j, :2].astype(np.int32)
          ret['hp_offset'][k * num_joints + j] = pts[j, :2] - pt_int
          ret['hp_ind'][k * num_joints + j] = \
            pt_int[1] * self.opt.output_w + pt_int[0]
          ret['hp_offset_mask'][k * num_joints + j] = 1
          ret['hm_hp_mask'][k * num_joints + j] = 1
          ret['joint'][k * num_joints + j] = j
          draw_umich_gaussian(
            ret['hm_hp'][j], pt_int, hp_radius)
          if pts[j, 2] == 1:
            ret['hm_hp'][j, pt_int[1], pt_int[0]] = self.ignore_val
            ret['hp_offset_mask'][k * num_joints + j] = 0
            ret['hm_hp_mask'][k * num_joints + j] = 0
        else:
          pts[j, :2] *= 0
      else:
        pts[j, :2] *= 0
        self._ignore_region(
          ret['hm_hp'][j, int(bbox[1]): int(bbox[3]) + 1, 
                          int(bbox[0]): int(bbox[2]) + 1])
    gt_det['hps'].append(pts[:, :2].reshape(num_joints * 2))

  def _add_rot(self, ret, ann, k, gt_det):
    if 'alpha' in ann:
      ret['rot_mask'][k] = 1
      alpha = ann['alpha']
      if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
        ret['rotbin'][k, 0] = 1
        ret['rotres'][k, 0] = alpha - (-0.5 * np.pi)    
      if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
        ret['rotbin'][k, 1] = 1
        ret['rotres'][k, 1] = alpha - (0.5 * np.pi)
      gt_det['rot'].append(self._alpha_to_8(ann['alpha']))
    else:
      gt_det['rot'].append(self._alpha_to_8(0))
    
  def _alpha_to_8(self, alpha):
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
  
  def _format_gt_det(self, gt_det):
    if (len(gt_det['scores']) == 0):
      gt_det = {'bboxes': np.array([[0,0,1,1]], dtype=np.float32), 
                'scores': np.array([1], dtype=np.float32), 
                'clses': np.array([0], dtype=np.float32),
                'cts': np.array([[0, 0]], dtype=np.float32),
                'pre_cts': np.array([[0, 0]], dtype=np.float32),
                'tracking': np.array([[0, 0]], dtype=np.float32),
                'bboxes_amodal': np.array([[0, 0]], dtype=np.float32),
                'hps': np.zeros((1, 17, 2), dtype=np.float32),}
    gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
    return gt_det

  def fake_video_data(self):
    self.coco.dataset['videos'] = []
    for i in range(len(self.coco.dataset['images'])):
      img_id = self.coco.dataset['images'][i]['id']
      self.coco.dataset['images'][i]['video_id'] = img_id
      self.coco.dataset['images'][i]['frame_id'] = 1
      self.coco.dataset['videos'].append({'id': img_id})
    
    if not ('annotations' in self.coco.dataset):
      return

    for i in range(len(self.coco.dataset['annotations'])):
      self.coco.dataset['annotations'][i]['track_id'] = i + 1
