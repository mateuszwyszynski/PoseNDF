"""Sample poses from manifold"""
import numpy as np
from pathlib import Path
from pytorch3d.transforms import quaternion_to_axis_angle
import torch
import tyro

from configs.config import load_config
from experiments.body_model import BodyModel
from experiments.projection_algorithm import Projector
from experiments.exp_utils import visualize
from model.posendf import PoseNDF


def sample_poses(config: Path, ckpt_path: Path, num_poses: int = 1, poses_file: Path = None, out_dir: Path = None):
    opt = load_config(config)
    net = PoseNDF(opt)
    device= 'cuda:0'
    net.load_checkpoint_from_path(ckpt_path, device=device, training=False)
    if poses_file is None:
         #if noisy pose path not given, then generate random quaternions
        noisy_poses = torch.rand((num_poses,21,4))
        noisy_poses = torch.nn.functional.normalize(noisy_poses,dim=2).to(device=device)
    else:
        noisy_poses = np.load(poses_file)['pose']
        #randomly slect according to batch size
        subsample_indices = np.random.randint(0, len(noisy_poses), num_poses)
        noisy_poses = noisy_poses[subsample_indices]
        #apply flip
        noisy_poses = torch.from_numpy(noisy_poses.astype(np.float32)).to(device=device)
    #  load body model
    bm_dir_path = 'smpl/models'
    body_model = BodyModel(bm_path=bm_dir_path, model_type='smpl', batch_size=num_poses,  num_betas=10).to(device=device)

    # render initial noisy poses
    betas = torch.zeros((num_poses, 10)).to(device=device)
    noisy_poses_axis_angle = torch.zeros((len(noisy_poses), 23, 3)).to(device=device)
    noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
    smpl_init = body_model(betas=betas, pose_body=noisy_poses_axis_angle.view(-1, 69))
    visualize(smpl_init.vertices, smpl_init.faces, out_dir, device=device, render=True, prefix='init')

    # create Motion denoiser layer
    projector = Projector(net, out_path=out_dir)
    projected_poses_axis_angle = projector.project(noisy_poses)

    # render final poses
    smpl_init = body_model(betas=betas, pose_body=projected_poses_axis_angle.view(-1, 69))
    visualize(smpl_init.vertices, smpl_init.faces, out_dir, device=device, render=True, prefix='out')


if __name__ == '__main__':
    tyro.cli(sample_poses, description=__doc__)
