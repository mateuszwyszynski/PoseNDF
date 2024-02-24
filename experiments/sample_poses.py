"""Sample poses from manifold"""
import numpy as np
from pathlib import Path
import torch
import tyro

from configs.config import load_config
from experiments.body_model import BodyModel
from experiments.projection_algorithm import Projector
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

    # create Motion denoiser layer
    projector = Projector(net, body_model=body_model, batch_size=num_poses, out_path=out_dir)
    projector.project(noisy_poses)


if __name__ == '__main__':
    tyro.cli(sample_poses, description=__doc__)
