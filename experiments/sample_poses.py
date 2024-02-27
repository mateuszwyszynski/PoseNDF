"""Sample poses from manifold"""
import numpy as np
import os
from pathlib import Path
from pytorch3d.transforms import quaternion_to_axis_angle
import torch
import tyro

from configs.config import load_config
from experiments.body_model import BodyModel
from experiments.exp_utils import renderer
from model.posendf import PoseNDF


def sample_poses(
        config: str = 'config.yaml', ckpt_path: str = 'checkpoint_epoch_best.tar',
        num_poses: int = 1, poses_file: Path = None, max_projection_dist: float = 0.001,
        max_projection_steps: int = None, render: bool = False, save_projection_steps: bool = False
        ):
    opt = load_config(config)
    posendf = PoseNDF(opt)
    device= 'cuda:0'
    posendf.load_checkpoint_from_path(os.path.join(posendf.experiment_dir, ckpt_path), device=device, training=False)
    if poses_file is None:
        noisy_poses = torch.rand((num_poses,21,4))
        noisy_poses = torch.nn.functional.normalize(noisy_poses,dim=2).to(device=device)
    else:
        noisy_poses = np.load(poses_file)['pose']
        subsample_indices = np.random.randint(0, len(noisy_poses), num_poses)
        noisy_poses = noisy_poses[subsample_indices]
        noisy_poses = torch.from_numpy(noisy_poses.astype(np.float32)).to(device=device)

    # render initial noisy poses
    if render:
        bm_dir_path = 'smpl/models'
        body_model = BodyModel(bm_path=bm_dir_path, model_type='smpl', batch_size=num_poses,  num_betas=10).to(device=device)
        betas = torch.zeros((num_poses, 10)).to(device=device)
        noisy_poses_axis_angle = torch.zeros((len(noisy_poses), 23, 3)).to(device=device)
        noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
        smpl_init = body_model(betas=betas, pose_body=noisy_poses_axis_angle.view(-1, 69))
        renderer(smpl_init.vertices, smpl_init.faces, posendf.experiment_dir, device=device, prefix='init')

    projected_poses = posendf.project(
        noisy_poses, max_dist=max_projection_dist, max_steps=max_projection_steps,
        save_projection_steps=save_projection_steps
        )

    # render final poses
    if render:
        projected_poses_axis_angle = torch.zeros((len(projected_poses), 23, 3)).to(device=device)
        projected_poses_axis_angle[:, :21] = quaternion_to_axis_angle(projected_poses)
        smpl_init = body_model(betas=betas, pose_body=projected_poses_axis_angle.view(-1, 69))
        renderer(smpl_init.vertices, smpl_init.faces, posendf.experiment_dir, device=device, prefix='out')


if __name__ == '__main__':
    tyro.cli(sample_poses, description=__doc__)
