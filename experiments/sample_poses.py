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


class SamplePose(object):
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


    def project(self, noisy_poses, iterations=100):
        # create initial SMPL joints and vertices for visualition(to be used for data term)
        noisy_poses_axis_angle = torch.zeros((len(noisy_poses), 23, 3)).to(device=self.device)
        noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
        smpl_init = self.body_model(betas=self.betas, pose_body=noisy_poses_axis_angle.view(-1, 69))
        self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, device=self.device, joints=smpl_init.Jtr, render=True, prefix='init')

        noisy_poses, _ = quat_flip(noisy_poses)
        noisy_poses = torch.nn.functional.normalize(noisy_poses,dim=-1)

        noisy_poses.requires_grad = True
        
        for _ in range(iterations):
            noisy_poses = self.projection_step(noisy_poses)

        # create final results
        noisy_poses_axis_angle = torch.zeros((len(noisy_poses), 23, 3)).to(device=self.device)
        noisy_poses_axis_angle[:, :21] = quaternion_to_axis_angle(noisy_poses)
        smpl_init = self.body_model(betas=self.betas, pose_body=noisy_poses_axis_angle.view(-1, 69))
        self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, device=self.device, joints=smpl_init.Jtr, render=True,prefix='out')


def sample_pose(opt, ckpt, motion_file=None,gt_data=None, out_path=None):
    ### load the model
    net = PoseNDF(opt)
    device= 'cuda:0'
    ckpt = torch.load(ckpt, map_location='cpu')['model_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    net = net.to(device)
    batch_size= 10
    if motion_file is None:
         #if noisy pose path not given, then generate random quaternions
        noisy_pose = torch.rand((batch_size,21,4))
        noisy_pose = torch.nn.functional.normalize(noisy_pose,dim=2).to(device=device)
    else:
        noisy_pose = np.load(motion_file)['pose']
        #randomly slect according to batch size
        subsample_indices = np.random.randint(0, len(noisy_pose), batch_size)
        noisy_pose = noisy_pose[subsample_indices]
        #apply flip
        noisy_pose = torch.from_numpy(noisy_pose.astype(np.float32)).to(device=device)
    #  load body model
    bm_dir_path = 'smpl/models'
    body_model = BodyModel(bm_path=bm_dir_path, model_type='smpl', batch_size=batch_size,  num_betas=10).to(device=device)

    # create Motion denoiser layer
    pose_sampler = SamplePose(net, body_model=body_model, batch_size=batch_size, out_path=out_path)
    pose_sampler.project(noisy_pose)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interpolate using PoseNDF.'
    )
    parser.add_argument('--config', '-c', default='posendf/version2/config.yaml', type=str, help='Path to config file.')
    parser.add_argument('--ckpt_path', '-ckpt', default='posendf/version2/checkpoint_epoch_best.tar', type=str, help='Path to pretrained model.')
    parser.add_argument('--noisy_pose', '-np', nargs='?', type=str, const='training_data/ACCAD/Female1General_c3d.npz', help='Path to noisy motion file')
    parser.add_argument('--outpath_folder', '-out', default='posendf/version2', type=str, help='Path to output')
    args = parser.parse_args()

    opt = load_config(args.config)
    print(args.ckpt_path)
    sample_pose(opt, args.ckpt_path, motion_file=args.noisy_pose, out_path=args.outpath_folder)
