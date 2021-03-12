import numpy
from pprint import pprint
import os
import copy

_EPS = numpy.finfo(float).eps * 4.0

def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = numpy.array(l[4:8], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)
    q *= numpy.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)

def associate(first_keys, second_keys, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """

    first_keys = copy.deepcopy(first_keys)
    second_keys = copy.deepcopy(second_keys)

    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

dataset_type = 'TUM' # 'KITTI'

# input_name = 'rgbd_dataset_freiburg2_desk_secret'
input_name = 'rgbd_dataset_freiburg2_desk'
# input_name = 'rgbd_dataset_freiburg3_large_cabinet_validation'
# input_name = 'rgbd_dataset_freiburg3_structure_texture_far_validation'
# input_name = 'rgbd_dataset_freiburg3_structure_notexture_far_validation'

file_gt = f'/nethome/hkwon64/Datasets/public/TUM/{input_name}/groundtruth.txt'

file_save_gt = f'/coc/pcba1/hkwon64/imuTube/repos_v2/odometry/TrianFlow/demo_{dataset_type}/{input_name}_gt.txt'

file = open(file_gt)
data = file.read()
lines = data.replace(","," ").replace("\t"," ").split("\n") 
list = [[float(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
# pprint (list)

time_gt = {}
f = open(file_save_gt, 'w')
for i, l in enumerate(list):
  # convert from [t|q] to [R|t]
  # print (i, l)
  # assert False
  traj_mat = transform44(l[0:])
  # print (traj_mat)
  traj_mat = traj_mat[:3]
  # print (traj_mat)
  # assert False

  ts = l[0]

  # save
  pose = traj_mat.flatten()[:12] # [3x4]
  time_gt[ts] = pose

  line = " ".join([str(j) for j in pose])
  f.write(line + '\n')
f.close()
print('Trajectory Saved.')
# pprint (time_gt)

# associate timestep
dir_rgb = f'/nethome/hkwon64/Datasets/public/TUM/{input_name}/rgb'
list_rgb = os.listdir(dir_rgb)
list_rgb = [float(item[:-4]) for item in list_rgb if '.png' in item]
# print (list_rgb[0], list_rgb[0][:-4])
# pprint (list_rgb[0])

offset = 0.0
max_difference = 0.02

first_keys = [item[0] for item in list]
second_keys = list_rgb
# pprint (first_keys)
# pprint (second_keys)

file_pred = f'/coc/pcba1/hkwon64/imuTube/repos_v2/odometry/TrianFlow/demo_{dataset_type}/{input_name}.txt'
f = open(file_pred, 'r')
s = f.readlines()
f.close()
# print (len(s), len(list_rgb))
# assert False

time_pred = {}
for ts , line in zip(list_rgb, s):
  line_split = [float(i) for i in line.split(" ")]
  # print (ts, line_split)
  time_pred[ts] = line_split
# pprint (time_pred)

matches = associate(first_keys, second_keys, offset, max_difference)
# pprint (matches)
# print (len(matches), len(list_rgb))

file_save_gt_as = f'/coc/pcba1/hkwon64/imuTube/repos_v2/odometry/TrianFlow/demo_{dataset_type}/{input_name}_gt_associate.txt'

file_save_pred_as = f'/coc/pcba1/hkwon64/imuTube/repos_v2/odometry/TrianFlow/demo_{dataset_type}/{input_name}_associate.txt'

f_gt_as = open(file_save_gt_as, 'w')
f_pred_as = open(file_save_pred_as, 'w')

for ts_gt, ts_pred in matches:
  # print (ts_gt, ts_pred)

  pose_gt = time_gt[ts_gt]
  line_gt = " ".join([str(j) for j in pose_gt]) 
  f_gt_as.write(line_gt + '\n')
  # print (ts_gt, line_gt)

  withIdx = int(len(pose_gt) == 13)
  P = numpy.eye(4)
  for row in range(3):
    for col in range(4):
      P[row, col] = pose_gt[row*4 + col + withIdx]
  print (ts_gt)
  print (P)

  pose_pred = time_pred[ts_pred]
  line_pred = " ".join([str(j) for j in pose_pred])  
  f_pred_as.write(line_pred + '\n')
  # print (ts_pred, line_pred)

  withIdx = int(len(pose_pred) == 13)
  P = numpy.eye(4)
  for row in range(3):
    for col in range(4):
      P[row, col] = pose_pred[row*4 + col + withIdx]
  print (ts_pred)
  print (P)

  print ('--------------')
f_gt_as.close()
f_pred_as.close()
print ('save in ...', file_save_gt_as)
print ('save in ...', file_save_pred_as)
