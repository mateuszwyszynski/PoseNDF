"""
Projection algorithm for projecting onto the manifold of plausible poses.
"""

import numpy as np
import os
import torch
from torch.autograd import grad

from experiments.exp_utils import quat_flip


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
    def __init__(self, posendf, out_path='./experiment_results/sample_pose', debug=False, device='cuda:0'):
        self.debug = debug
        self.device = device
        self.pose_prior = posendf
        self.out_path = out_path


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
        noisy_poses, _ = quat_flip(noisy_poses)
        noisy_poses = torch.nn.functional.normalize(noisy_poses,dim=-1)

        noisy_poses.requires_grad = True

        if save_projection_steps:
            projection_steps_path = os.path.join(self.out_path, 'projection_steps')
            os.makedirs(projection_steps_path, exist_ok=True)
            batch_size, _, _ = noisy_poses.shape
            projection_steps = torch.detach(noisy_poses).reshape(batch_size, 1, -1, 4)
        
        for _ in range(iterations):
            noisy_poses = self.projection_step(noisy_poses)

            if save_projection_steps:
                projection_steps = torch.cat((projection_steps, noisy_poses.reshape(batch_size, 1, -1, 4)), dim=1)

        if save_projection_steps:
            for motion_ind in range(batch_size):
                np.savez(os.path.join(projection_steps_path, f'{motion_ind}.npz'), pose_body=np.array(projection_steps[motion_ind].cpu().detach().numpy()))

        return noisy_poses