"""
Projection algorithm for projecting onto the manifold of plausible poses.
"""

import numpy as np
import os
from pytorch3d.transforms import quaternion_to_axis_angle
from pytorch3d.io import save_obj
import torch
from torch.autograd import grad

from experiments.exp_utils import renderer, quat_flip


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
    def __init__(self, posendf, body_model, out_path='./experiment_results/sample_pose', debug=False, device='cuda:0', batch_size=1, gender='male'):
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
            batch_size, _, _ = noisy_poses_axis_angle.shape
            projection_steps = torch.detach(noisy_poses_axis_angle).reshape(batch_size, 1, -1)
        
        for _ in range(iterations):
            noisy_poses = self.projection_step(noisy_poses)

            if save_projection_steps:
                noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
                projection_steps = torch.cat((projection_steps, noisy_poses_axis_angle.reshape(batch_size, 1, -1)), dim=1)

        for motion_ind in range(batch_size):
            np.savez(os.path.join(projection_steps_path, f'{motion_ind}.npz'), pose_body=np.array(projection_steps[motion_ind].cpu().detach().numpy()))

        # create final results
        noisy_poses_axis_angle = torch.zeros((len(noisy_poses), 23, 3)).to(device=self.device)
        noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
        smpl_init = self.body_model(betas=self.betas, pose_body=noisy_poses_axis_angle.view(-1, 69))
        self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, device=self.device, joints=smpl_init.Jtr, render=True,prefix='out')