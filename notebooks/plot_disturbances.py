# %%
import numpy as np
from pathlib import Path
from pytorch3d.transforms import axis_angle_to_quaternion, standardize_quaternion
import torch

# %%
# Add the root directory to the path to import custom modules
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# %%
from data.dist_utils import pose_distance

# %% Load poses
poses1_fpath = '../amass_raw/ACCAD/Female1General_c3d/A6- lift box t2_poses.npz'
poses2_fpath = '../amass-disturbed/ACCAD/Female1General_c3d/A6- lift box t2_poses.npz'

poses1 = np.load(poses1_fpath)['poses']
poses2 = np.load(poses2_fpath)['poses']

# %% Convert poses to quaternions
body_angles1 = torch.from_numpy(poses1[:, 3:66].reshape(-1, 21, 3))
body_quats1 = standardize_quaternion(axis_angle_to_quaternion(body_angles1))

body_angles2 = torch.from_numpy(poses2[:, 3:66].reshape(-1, 21, 3))
body_quats2 = standardize_quaternion(axis_angle_to_quaternion(body_angles2))

# %% Calculate distances
distances = pose_distance(body_quats1, body_quats2)

# %% Plot distances
import matplotlib.pyplot as plt

plt.plot(distances)
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.title('Distances between original and disturbed poses')
plt.show()