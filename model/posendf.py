from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
import ipdb
#todo: create a base trainer
from model.network.net_modules import  StructureEncoder, DFNet
from experiments.exp_utils import quat_flip

import time
import pickle as pkl
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


class PoseNDF(torch.nn.Module):

    def __init__(self, opt):
        super(PoseNDF, self).__init__()

        self.device = opt['train']['device']

        # create all the models:
        # self.shape_model = ShapeNet().to(self.device)
        # self.pose_model = PoseNet().to(self.device)
        self.enc = None
        if opt['model']['StrEnc']['use']:
            self.enc = StructureEncoder(opt['model']['StrEnc']).to(self.device)

        self.dfnet = DFNet(opt['model']['DFNet']).to(self.device)
        
        
        #geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))  todo: do we need this???
        self.loss = opt['train']['loss_type']
        self.batch_size= opt['train']['batch_size']


        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()

        self.experiment_dir = opt['experiment']['root_dir']


    def load_checkpoint_from_path(self, checkpoint_path, device, training=False):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        if training:
            self.train()
        else:
            self.eval()
        self = self.to(device)
        
       
    def train(self, mode=True):
        super().train(mode)


    def forward(self, pose, dist_gt=None, man_poses=None, train=True,eikonal=0.0 ):

        pose = pose.to(device=self.device).reshape(-1,21,4)
        if train and eikonal > 0.0:
            pose.requires_grad=True
        #pose = torch.nn.functional.normalize(pose.to(device=self.device),dim=-1)

        if train:
            dist_gt = dist_gt.to(device=self.device).reshape(-1)
        if self.enc:
            rand_pose_in = self.enc(pose)
        dist_pred = self.dfnet(rand_pose_in)
        if train:
            #calculate distance for manifold poses
            man_poses = man_poses.to(device=self.device).reshape(-1,21,4)
            #man_poses = torch.nn.functional.normalize(man_poses.to(device=self.device),dim=-1)
            if self.enc:
                man_pose_in = self.enc(man_poses)
            dist_man = self.dfnet(man_pose_in)
            loss = self.loss_l1(dist_pred[:,0], dist_gt)
            loss_man = (dist_man.abs()).mean()
            if eikonal > 0.0:
                # eikonal term loss
                grad_val = gradient(pose, dist_pred)
                eikonal_loss =  ((grad_val.norm(2, dim=-1) - 1) ** 2).mean()
                return loss, {'dist': loss , 'man_loss': loss_man, 'eikonal': eikonal_loss}

            return loss, {'dist': loss,  'man_loss': loss_man }
        else:
            return {'dist_pred': dist_pred}


    def projection_step(self, poses):
        net_pred = self(poses, train=False)
        grad_val = gradient(poses, net_pred['dist_pred']).reshape(-1, 84)
        poses = poses.detach()
        net_pred['dist_pred'] = net_pred['dist_pred'].detach()
        grad_norm = torch.nn.functional.normalize(grad_val, p=2.0, dim=-1)
        poses = poses - (net_pred['dist_pred']*grad_norm).reshape(-1, 21,4)
        poses, _ = quat_flip(poses)
        poses = torch.nn.functional.normalize(poses,dim=-1)
        poses = poses.detach()
        poses.requires_grad = True

        return poses


    def project(self, poses, iterations=100, save_projection_steps=True):
        poses, _ = quat_flip(poses)
        poses = torch.nn.functional.normalize(poses,dim=-1)

        poses.requires_grad = True

        if save_projection_steps:
            projection_steps_path = os.path.join(self.experiment_dir, 'projection_steps')
            os.makedirs(projection_steps_path, exist_ok=True)
            batch_size, _, _ = poses.shape
            projection_steps = torch.detach(poses).reshape(batch_size, 1, -1, 4)

        for _ in range(iterations):
            poses = self.projection_step(poses)

            if save_projection_steps:
                projection_steps = torch.cat((projection_steps, poses.reshape(batch_size, 1, -1, 4)), dim=1)

        if save_projection_steps:
            for motion_ind in range(batch_size):
                np.savez(os.path.join(projection_steps_path, f'{motion_ind}.npz'), pose_body=np.array(projection_steps[motion_ind].cpu().detach().numpy()))

        return poses
