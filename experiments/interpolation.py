import argparse
from configs.config import load_config
# General config

#from model_quat import  train_manifold2 as train_manifold
from model.posendf import PoseNDF
import shutil
from data.data_splits import amass_splits
import ipdb
import torch
import numpy as np
import os
import tyro


def main(
        config: str = 'posendf/replicate-version2/config.yaml',
        ckpt: str = 'posendf/replicate-version2/checkpoint_epoch_best.tar',
        num_iter: int = 20, step_size: float = 0.1, poses_file: str = None, save_interpolation_steps: bool = False
        ) -> None:
    opt = load_config(config)
    posendf = PoseNDF(opt)
    device= 'cuda:0'
    ckpt = torch.load(ckpt, map_location='cpu')['model_state_dict']
    posendf.load_state_dict(ckpt)
    posendf.eval()
    posendf = posendf.to(device)

    if poses_file is None:
        pose1 = np.random.rand(21,4).astype(np.float32)
        pose2 = np.random.rand(21,4).astype(np.float32)
    else:
        noisy_poses = np.load(poses_file)['pose']
        start_ind = 0
        pose1 = noisy_poses[start_ind]
        pose2 = noisy_poses[len(noisy_poses)-1]
    pose1 = torch.from_numpy(pose1.astype(np.float32)).unsqueeze(0)
    pose2 = torch.from_numpy(pose2.astype(np.float32)).unsqueeze(0)

    pose1 = posendf.project(pose1.to(device))
    pose2 = posendf.project(pose2.to(device))

    posendf.interpolate(pose1, pose2, num_iter, step_size, save_interpolation_steps=save_interpolation_steps)



if __name__ == '__main__':
    tyro.cli(main, description=__doc__)