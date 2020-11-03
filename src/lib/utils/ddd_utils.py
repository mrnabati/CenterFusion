from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.spatial import ConvexHull
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

import numpy as np
import cv2

def comput_corners_3d(dim, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners).transpose(1, 0)
  return corners_3d

def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  corners_3d = comput_corners_3d(dim, rotation_y)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
  return corners_3d

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

def draw_box_3d(image, corners, c=(255, 0, 255), same_color=False):
  face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [3,0,4,7],
              [2,3,7,6]]
  right_corners = [1, 2, 6, 5] if not same_color else []
  left_corners = [0, 3, 7, 4] if not same_color else []
  thickness = 4 if same_color else 2
  corners = corners.astype(np.int32)
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      # print('corners', corners)
      cc = c
      if (f[j] in left_corners) and (f[(j+1)%4] in left_corners):
        cc = (255, 0, 0)
      if (f[j] in right_corners) and (f[(j+1)%4] in right_corners):
        cc = (0, 0, 255)
      try:
        cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
            (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), cc, thickness, lineType=cv2.LINE_AA)
      except:
        pass
    if ind_f == 0:
      try:
        cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                 (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                 (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
      except:
        pass
    # top_idx = [0, 1, 2, 3]
  return image

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32).reshape(3)
  return pt_3d

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot(center, alpha, dim, depth, calib):
  # single image
  locations = unproject_2d_to_3d(center, depth, calib)
  locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
  box_3d = compute_box_3d(dim, location, rotation_y)
  box_2d = project_to_image(box_3d, calib)
  return box_2d

#-----------------------------------------------------
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def iou3d(corners1, corners2):
  ''' Compute 3D bounding box IoU.
  Input:
      corners1: numpy array (8,3), assume up direction is negative Y
      corners2: numpy array (8,3), assume up direction is negative Y
  Output:
      iou: 3D bounding box IoU
      iou_2d: bird's eye view 2D bounding box IoU
  '''
  # corner points are in counter clockwise order
  rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
  rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
  area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
  area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
  inter, inter_area = convex_hull_intersection(rect1, rect2)
  iou_2d = inter_area/(area1+area2-inter_area)
  ymax = min(corners1[0,1], corners2[0,1])
  ymin = max(corners1[4,1], corners2[4,1])
  inter_vol = inter_area * max(0.0, ymax-ymin)
  vol1 = box3d_vol(corners1)
  vol2 = box3d_vol(corners2)
  iou = inter_vol / (vol1 + vol2 - inter_vol)
  return iou, iou_2d


def iou3d_global(corners1, corners2):
  ''' Compute 3D bounding box IoU.
  Input:
      corners1: numpy array (8,3), assume up direction is negative Y
      corners2: numpy array (8,3), assume up direction is negative Y
  Output:
      iou: 3D bounding box IoU
      iou_2d: bird's eye view 2D bounding box IoU
  '''
  # corner points are in counter clockwise order
  rect1 = corners1[:,[0,3,7,4]].T
  rect2 = corners2[:,[0,3,7,4]].T
  
  rect1 = [(rect1[i,0], rect1[i,1]) for i in range(3,-1,-1)]
  rect2 = [(rect2[i,0], rect2[i,1]) for i in range(3,-1,-1)]
  
  area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
  area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
  inter, inter_area = convex_hull_intersection(rect1, rect2)

  iou_2d = inter_area/(area1+area2-inter_area)
  
  iou = 0
  # ymax = min(corners1[0,2], corners2[0,2])
  # ymin = max(corners1[1,2], corners2[1,2])
  # inter_vol = inter_area * max(0.0, ymax-ymin)
  # vol1 = box3d_vol(corners1)
  # vol2 = box3d_vol(corners2)
  # iou = inter_vol / (vol1 + vol2 - inter_vol)
  return iou, iou_2d


def get_pc_hm(pc_hm, pc_dep, dep, bbox, dist_thresh, opt):
  if len(dep) > 0:
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
    nonzero_pc_dep = np.exp(-pc_dep[nonzero_inds])
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




if __name__ == '__main__':
  calib = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  alpha = -0.20
  tl = np.array([712.40, 143.00], dtype=np.float32)
  br = np.array([810.73, 307.92], dtype=np.float32)
  ct = (tl + br) / 2
  rotation_y = 0.01
  print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
  print('rotation_y', rotation_y)
  
  
  # Test iou3d (camera coord)
  box_1 = compute_box_3d(dim=[1.599275 , 1.9106505, 4.5931444],
                         location=[ 7.1180778,  2.1364648, 41.784885],
                         rotation_y= -1.1312813047259618)
  box_2 = compute_box_3d(dim=[1.599275 , 1.9106505, 4.5931444],
                         location=[ 7.1180778,  2.1364648, 41.784885],
                         rotation_y= -1.1312813047259618)
  iou = iou3d(box_1, box_2)
  print("Results should be almost 1.0: ", iou)
  
  # # Test iou3d (global coord)  
  translation1 = [634.7540893554688, 1620.952880859375, 0.4360223412513733]
  size1 = [1.9073231220245361, 4.5971598625183105, 1.5940513610839844]
  rotation1 = [-0.6379619591303222, 0.6256341359192967, -0.320485847319929, 0.31444441216651253]
  
  translation2 = [634.7540893554688, 1620.952880859375, 0.4360223412513733]
  size2 = [1.9073231220245361, 4.5971598625183105, 1.5940513610839844]
  rotation2 = [-0.6379619591303222, 0.6256341359192967, -0.320485847319929, 0.31444441216651253]
  
  box_1 = Box(translation1, size1, Quaternion(rotation1))
  box_2 = Box(translation2, size2, Quaternion(rotation2))
  iou, iou_2d = iou3d_global(box_1.corners(), box_2.corners())
  print(iou, iou_2d)
  
