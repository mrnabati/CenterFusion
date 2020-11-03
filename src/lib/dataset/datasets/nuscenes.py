# Copyright (c) Xingyi Zhou. All Rights Reserved
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from pyquaternion import Quaternion
import numpy as np
import torch
import json
import cv2
import os
import math
import copy
from tqdm import tqdm

from ..generic_dataset import GenericDataset
from utils.ddd_utils import compute_box_3d, project_to_image, iou3d_global
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from itertools import compress

class nuScenes(GenericDataset):
  default_resolution = [448, 800]
  num_categories = 10
  class_name = [
    'car', 'truck', 'bus', 'trailer', 
    'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
    'traffic_cone', 'barrier']
  cat_ids = {i + 1: i + 1 for i in range(num_categories)}
  focal_length = 1200
  max_objs = 128
  _tracking_ignored_class = ['construction_vehicle', 'traffic_cone', 'barrier']
  _vehicles = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']
  _cycles = ['motorcycle', 'bicycle']
  _pedestrians = ['pedestrian']
  attribute_to_id = {
    '': 0, 'cycle.with_rider' : 1, 'cycle.without_rider' : 2,
    'pedestrian.moving': 3, 'pedestrian.standing': 4, 
    'pedestrian.sitting_lying_down': 5,
    'vehicle.moving': 6, 'vehicle.parked': 7, 
    'vehicle.stopped': 8}
  id_to_attribute = {v: k for k, v in attribute_to_id.items()}


  def __init__(self, opt, split):
    split_names = {
        'mini_train':'mini_train', 
        'mini_val':'mini_val',
        'train': 'train', 
        'train_detect': 'train_detect',
        'train_track':'train_track', 
        'val': 'val',
        'test': 'test',
        'mini_train_2': 'mini_train_2',
        'trainval': 'trainval',
    }
    
    split_name = split_names[split]
    data_dir = os.path.join(opt.data_dir, 'nuscenes')
    print('Dataset version', opt.dataset_version)
    
    anns_dir = 'annotations'
    if opt.radar_sweeps > 1:
      anns_dir += '_{}sweeps'.format(opt.radar_sweeps)

    if opt.dataset_version == 'test':
      ann_path = os.path.join(data_dir, anns_dir, 'test.json')
    else:
      ann_path = os.path.join(data_dir, anns_dir, '{}.json').format(split_name)

    self.images = None
    super(nuScenes, self).__init__(opt, split, ann_path, data_dir)

    self.alpha_in_degree = False    
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))


  def __len__(self):
    return self.num_samples


  def _to_float(self, x):
    return float("{:.2f}".format(x))


  def convert_coco_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      if type(all_bboxes[image_id]) != type({}):
        # newest format
        for j in range(len(all_bboxes[image_id])):
          item = all_bboxes[image_id][j]   
          category_id = citem['class']
          bbox = item['bbox']
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(item['score']))
          }
          detections.append(detection)
    return detections


  def convert_eval_format(self, results):
    ret = {'meta': {'use_camera': True, 'use_lidar': False, 
      'use_radar': self.opt.pointcloud, 
      'use_map': False, 'use_external': False}, 'results': {}}
    print('Converting nuscenes format...')
    for image_id in self.images:
      if not (image_id in results):
        continue
      image_info = self.coco.loadImgs(ids=[image_id])[0]
      sample_token = image_info['sample_token']
      trans_matrix = np.array(image_info['trans_matrix'], np.float32)
      velocity_mat = np.array(image_info['velocity_trans_matrix'], np.float32)
      sensor_id = image_info['sensor_id']
      sample_results = []
      for item in results[image_id]:
        class_name = self.class_name[int(item['class'] - 1)] \
            if not ('detection_name' in item) else item['detection_name']
        if self.opt.tracking and class_name in self._tracking_ignored_class:
          continue
        score = float(item['score']) \
            if not ('detection_score' in item) else item['detection_score']
        if 'size' in item:
          size = item['size']
        else:
          size = [float(item['dim'][1]), float(item['dim'][2]), \
            float(item['dim'][0])]
        if 'translation' in item:
          translation = item['translation']
        else:
          translation = np.dot(trans_matrix, np.array(
            [item['loc'][0], item['loc'][1] - size[2], item['loc'][2], 1], 
            np.float32))

        det_id = item['det_id'] if 'det_id' in item else -1
        tracking_id = item['tracking_id'] if 'tracking_id' in item else 1
        
        if not ('rotation' in item):
          rot_cam = Quaternion(
            axis=[0, 1, 0], angle=item['rot_y'])
          loc = np.array(
            [item['loc'][0], item['loc'][1], item['loc'][2]], np.float32)
          box = Box(loc, size, rot_cam, name='2', token='1')
          box.translate(np.array([0, - box.wlh[2] / 2, 0]))
          box.rotate(Quaternion(image_info['cs_record_rot']))
          box.translate(np.array(image_info['cs_record_trans']))
          box.rotate(Quaternion(image_info['pose_record_rot']))
          box.translate(np.array(image_info['pose_record_trans']))
          rotation = box.orientation
          rotation = [float(rotation.w), float(rotation.x), \
            float(rotation.y), float(rotation.z)]
        else:
           rotation = item['rotation']
        
        nuscenes_att = np.array(item['nuscenes_att'], np.float32) \
          if 'nuscenes_att' in item else np.zeros(8, np.float32)
        att = ''
        if class_name in self._cycles:
          att = self.id_to_attribute[np.argmax(nuscenes_att[0:2]) + 1]
        elif class_name in self._pedestrians:
          att = self.id_to_attribute[np.argmax(nuscenes_att[2:5]) + 3]
        elif class_name in self._vehicles:
          att = self.id_to_attribute[np.argmax(nuscenes_att[5:8]) + 6]
        if 'velocity' in item and len(item['velocity']) == 2:
          velocity = item['velocity']
        else:
          velocity = item['velocity'] if 'velocity' in item else [0, 0, 0]
          
          velocity = np.dot(velocity_mat, np.array(
            [velocity[0], velocity[1], velocity[2], 0], np.float32))
          # velocity = np.dot(trans_matrix, np.array(
          #   [velocity[0], velocity[1], velocity[2], 0], np.float32))
          velocity = [float(velocity[0]), float(velocity[1])]

        result = {
          'sample_token': sample_token, 
          'translation': [float(translation[0]), float(translation[1]), \
            float(translation[2])],
          'size': size,
          'rotation': rotation,
          'velocity': velocity,
          'detection_name': class_name,
          'attribute_name': att \
            if not ('attribute_name' in item) else item['attribute_name'],
          'detection_score': score,
          'tracking_name': class_name,
          'tracking_score': score,
          'tracking_id': tracking_id,
          'sensor_id': sensor_id,
          'det_id': det_id}

        sample_results.append(result)
      if sample_token in ret['results']:
        ret['results'][sample_token] = ret['results'][sample_token] + \
          sample_results
      else:
        ret['results'][sample_token] = sample_results

    for sample_token in ret['results'].keys():
      confs = sorted([(-d['detection_score'], ind) \
        for ind, d in enumerate(ret['results'][sample_token])])
      ret['results'][sample_token] = [ret['results'][sample_token][ind] \
        for _, ind in confs[:min(500, len(confs))]]
    
    if self.opt.iou_thresh > 0:
      print("Applying BEV NMS...")
      n_removed = 0
      for sample_token, dets in tqdm(ret['results'].items()):
        ret['results'][sample_token], n = self.apply_bev_nms(dets, self.opt.iou_thresh, 
                                                          dist_thresh=2)
        n_removed += n
      print("Removed {} detections with IOU > {}".format(n_removed, self.opt.iou_thresh))

    return ret


  def apply_bev_nms(self, dets, iou_thresh, dist_thresh=2):
    """
    Filter detection results in every sample based on BEV IOU of bounding boxes.
    results in each sample must be sorted by score

    Ouput:
      ious: list of ious
      n: number of remove detections
    """
    N = len(dets)
    ious = []
    for i in range(N):

      try:
        ious = self.bev_iou(dets[i], dets[i+1:])
      except (ValueError, IndexError) as e:
        break
      
      iou_mask = (np.array(ious) < iou_thresh)
      dets = dets[:i+1] + list(compress(dets[i+1:], iou_mask))

    return dets, N-len(dets)
  

  def bev_iou(self, det1, det2, dist_thresh = 2):
    ious = []
    for det in det2:
      dist = np.linalg.norm(np.array(det1['translation'][:2]) - np.array(det['translation'][:2]))
      if dist > dist_thresh:
        ious.append(0)
        continue

      box1 = Box(det1['translation'], det1['size'], Quaternion(det1['rotation']))
      box2 = Box(det['translation'], det['size'], Quaternion(det['rotation']))
      iou, iou_2d = iou3d_global(box1.corners(), box2.corners())
      ious.append(iou_2d)
    
    return ious


  def save_results(self, results, save_dir, task, split):
    json.dump(self.convert_eval_format(results), 
                open('{}/results_nuscenes_{}_{}.json'.format(save_dir, task, split), 'w'))


  def run_eval(self, results, save_dir, n_plots=10, render_curves=False):
    task = 'tracking' if self.opt.tracking else 'det'
    split = self.opt.val_split
    version = 'v1.0-mini' if 'mini' in split else 'v1.0-trainval'
    self.save_results(results, save_dir, task, split)
    render_curves = 1 if render_curves else 0
    
    if task == 'det':
      output_dir = '{}/nuscenes_eval_det_output_{}/'.format(save_dir, split)
      os.system('python ' + \
        'tools/nuscenes-devkit/python-sdk/nuscenes/eval/detection/evaluate.py ' + \
        '{}/results_nuscenes_{}_{}.json '.format(save_dir, task, split) + \
        '--output_dir {} '.format(output_dir) + \
        '--eval_set {} '.format(split) + \
        '--dataroot ../data/nuscenes/ ' + \
        '--version {} '.format(version) + \
        '--plot_examples {} '.format(n_plots) + \
        '--render_curves {} '.format(render_curves))
    else:
      output_dir = '{}/nuscenes_evaltracl__output/'.format(save_dir)
      os.system('python ' + \
        'tools/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/evaluate.py ' + \
        '{}/results_nuscenes_{}_{}.json '.format(save_dir, task, split) + \
        '--output_dir {} '.format(output_dir) + \
        '--dataroot ../data/nuscenes/')
      os.system('python ' + \
        'tools/nuscenes-devkit/python-sdk-alpha02/nuscenes/eval/tracking/evaluate.py ' + \
        '{}/results_nuscenes_{}_{}.json '.format(save_dir, task, split) + \
        '--output_dir {} '.format(output_dir) + \
        '--dataroot ../data/nuscenes/')
    
    return output_dir
