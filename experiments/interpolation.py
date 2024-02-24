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
        num_iter: int = 20, step_size: float = 0.1, poses_file: str = None
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

    interpolation_steps_path = os.path.join(opt['experiment']['root_dir'], 'interpolation_steps')
    os.makedirs(interpolation_steps_path, exist_ok=True)
    interpolation_steps = torch.detach(pose1).reshape(1, -1, 4)

    for _ in range(num_iter):
        pose = (1-step_size)*pose1 + step_size*pose2
        pose1 = posendf.project(torch.detach(pose))
        interpolation_steps = torch.cat((interpolation_steps, pose1.reshape(1, -1, 4)), dim=0)

    np.savez(os.path.join(interpolation_steps_path, 'interpolation.npz'), pose_body=np.array(interpolation_steps.cpu().detach().numpy()))



if __name__ == '__main__':
    tyro.cli(main, description=__doc__)