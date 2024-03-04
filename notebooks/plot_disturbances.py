# %% This notebook uses Jupytext to store the notebook in .py format: https://github.com/mwouts/jupytext

# %%
import numpy as np
from pathlib import Path
from pytorch3d.transforms import axis_angle_to_quaternion, standardize_quaternion
import torch

# %% Add the root directory to the path to import custom modules
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# %%
from configs.config import load_config
from data.dist_utils import pose_distance
from model.posendf import PoseNDF

# %% Load poses
poses_fpath_suffix = 'ACCAD/Female1General_c3d/A6- lift box t2_poses.npz'
poses1_fpath = '../amass_raw/' + poses_fpath_suffix
poses2_fpath = '../amass-disturbed/' + poses_fpath_suffix

poses1 = np.load(poses1_fpath)['poses']
poses2 = np.load(poses2_fpath)['poses']

# %% Convert poses to quaternions
body_angles1 = torch.from_numpy(poses1[:, 3:66].reshape(-1, 21, 3).astype(np.float32))
body_quats1 = standardize_quaternion(axis_angle_to_quaternion(body_angles1))

body_angles2 = torch.from_numpy(poses2[:, 3:66].reshape(-1, 21, 3).astype(np.float32))
body_quats2 = standardize_quaternion(axis_angle_to_quaternion(body_angles2))

# %% Calculate distances between poses
distances = pose_distance(body_quats1, body_quats2)

# %% Load model
config_fpath = '../posendf/replicate-version2/config.yaml'
config = load_config(config_fpath)
posendf = PoseNDF(config)
device = 'cuda:0'
posendf.load_checkpoint_from_path(Path('../', posendf.experiment_dir, 'checkpoint_epoch_best.tar'), device=device, training=False)

# %% Compute distances to the manifold
manifold_distances1 = posendf(body_quats1, train=False)['dist_pred'].reshape(-1).detach().cpu()
manifold_distances2 = posendf(body_quats2, train=False)['dist_pred'].reshape(-1).detach().cpu()

# %% Expected upper and lower bounds on manifold distances
upper_bounds = manifold_distances1 + distances
lower_bounds = manifold_distances1 - distances

# %% [markdown]
# We expect that the distance to the manifold for the disturbed poses will change no more than the added disturbance

# %% Plot distances
import matplotlib.pyplot as plt

plt.plot(manifold_distances1, 'bo', label='Raw poses to manifold')
plt.plot(manifold_distances2, 'ro', markersize=3, label='Disturbed poses to manifold')
plt.plot(upper_bounds, 'go', markersize=1, label='Disturbance bounds')
plt.plot(lower_bounds, 'go', markersize=1)
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.title('Distances between original and disturbed poses')
plt.legend()
plt.show()

# %% Incorrect cases - when we exceed the upper bound
manifold_distances2[(manifold_distances2 - upper_bounds) > 0.0]

# %% Incorrect cases - when we fall below the lower bound
manifold_distances2[(lower_bounds - manifold_distances2) > 0.0]
