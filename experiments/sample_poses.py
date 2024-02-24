"""Sample poses from manifold"""

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
from experiments.body_model import BodyModel
from experiments.exp_utils import renderer, quat_flip

from pytorch3d.transforms import axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle
from tqdm import tqdm
from pytorch3d.io import save_obj
import os

from torch.autograd import grad
import tyro
from pathlib import Path


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


class Projector(object):
    def __init__(self, posendf,  body_model, out_path='./experiment_results/sample_pose', debug=False, device='cuda:0', batch_size=1, gender='male'):
        self.debug = debug
        self.device = device
        self.pose_prior = posendf
        self.body_model = body_model
        self.out_path = out_path
        self.betas = torch.zeros((batch_size,10)).to(device=self.device)  #for visualization
    
    @staticmethod
    def visualize(vertices, faces, out_path, device, joints=None, render=False, prefix='out', save_mesh=False):
        # save meshes and rendered results if needed
        os.makedirs(out_path,exist_ok=True)
        if save_mesh:
            os.makedirs( os.path.join(out_path, 'meshes'), exist_ok=True)
            [save_obj(os.path.join(out_path, 'meshes', '{}_{:04}.obj'.format(prefix,i) ), vertices[i], faces) for i in range(len(vertices))]

        if render:
            renderer(vertices, faces, out_path, device=device,  prefix=prefix)

    def projection_step(self, noisy_poses):
        net_pred = self.pose_prior(noisy_poses, train=False)
        grad_val = gradient(noisy_poses, net_pred['dist_pred']).reshape(-1, 84)
        noisy_poses = noisy_poses.detach()
        net_pred['dist_pred'] = net_pred['dist_pred'].detach()
        print(torch.mean(net_pred['dist_pred']))
        grad_norm = torch.nn.functional.normalize(grad_val, p=2.0, dim=-1)
        noisy_poses = noisy_poses - (net_pred['dist_pred']*grad_norm).reshape(-1, 21,4)
        noisy_poses, _ = quat_flip(noisy_poses)
        noisy_poses = torch.nn.functional.normalize(noisy_poses,dim=-1)
        noisy_poses = noisy_poses.detach()
        noisy_poses.requires_grad = True

        # print(torch.mean(net_pred['dist_pred']))
        # grad = gradient(noisy_poses, net_pred['dist_pred']).reshape(-1, 84)
        # grad_norm = torch.nn.functional.normalize(grad, p=2.0, dim=-1)
        # noisy_poses = noisy_poses - (net_pred['dist_pred']*grad_norm).reshape(-1, 21,4)
        # noisy_poses = torch.nn.functional.normalize(noisy_poses,dim=-1

        return noisy_poses


    def project(self, noisy_poses, iterations=100, save_projection_steps=True):
        # create initial SMPL joints and vertices for visualition(to be used for data term)
        noisy_poses_axis_angle = torch.zeros((len(noisy_poses), 23, 3)).to(device=self.device)
        noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
        smpl_init = self.body_model(betas=self.betas, pose_body=noisy_poses_axis_angle.view(-1, 69))
        self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, device=self.device, joints=smpl_init.Jtr, render=True, prefix='init')

        noisy_poses, _ = quat_flip(noisy_poses)
        noisy_poses = torch.nn.functional.normalize(noisy_poses,dim=-1)

        noisy_poses.requires_grad = True

        if save_projection_steps:
            projection_steps_path = os.path.join(self.out_path, 'projection_steps')
            os.makedirs(projection_steps_path, exist_ok=True)
            batch_size, num_joints, angles = noisy_poses_axis_angle.shape
            projection_steps = torch.clone(noisy_poses_axis_angle).reshape(batch_size, 1, num_joints*angles)
        
        for _ in range(iterations):
            noisy_poses = self.projection_step(noisy_poses)

            if save_projection_steps:
                noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
                projection_steps = torch.cat((projection_steps, noisy_poses_axis_angle.reshape(batch_size, 1, num_joints*angles)), dim=1)

        for motion_ind in range(batch_size):
            np.savez(os.path.join(projection_steps_path, f'{motion_ind}.npz'), pose_body=np.array(projection_steps[motion_ind].cpu().detach().numpy()))

        # create final results
        noisy_poses_axis_angle = torch.zeros((len(noisy_poses), 23, 3)).to(device=self.device)
        noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
        smpl_init = self.body_model(betas=self.betas, pose_body=noisy_poses_axis_angle.view(-1, 69))
        self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, device=self.device, joints=smpl_init.Jtr, render=True,prefix='out')


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
