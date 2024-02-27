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
        config: str = 'config.yaml', ckpt_path: str = 'checkpoint_epoch_best.tar', poses_file: str = None,
        num_steps: int = 20, step_size: float = 0.1, max_projection_dist: float = 0.001,
        max_projection_steps: int = None, save_interpolation_steps: bool = False
        ) -> None:
    opt = load_config(config)
    posendf = PoseNDF(opt)
    device= 'cuda:0'
    posendf.load_checkpoint_from_path(os.path.join(posendf.experiment_dir, ckpt_path), device=device, training=False)

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

    pose1 = posendf.project(pose1.to(device), max_dist=max_projection_dist, max_steps=max_projection_steps)
    pose2 = posendf.project(pose2.to(device), max_dist=max_projection_dist, max_steps=max_projection_steps)

    posendf.interpolate(
        pose1, pose2, num_steps, step_size, max_projection_dist=max_projection_dist,
        max_projection_steps=max_projection_steps, save_interpolation_steps=save_interpolation_steps
        )


if __name__ == '__main__':
    tyro.cli(main, description=__doc__)