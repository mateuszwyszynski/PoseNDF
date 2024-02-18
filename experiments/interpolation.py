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

def main(opt, ckpt):
    ### load the model
    net = PoseNDF(opt)
    device= 'cuda:0'
    ckpt = torch.load(ckpt, map_location='cpu')['model_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    net = net.to(device)
    ipdb.set_trace()
    pose1 = torch.from_numpy(np.random.rand(21,4).astype(np.float32)).unsqueeze(0)
    pose2 = torch.from_numpy(np.random.rand(21,4).astype(np.float32)).unsqueeze(0)
    tmp = net(pose1,train=False)
    ipdb.set_trace()
    # load the pose from npz file and convert pose in axis angle to quaternion
    # with torch.no_grad():




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interpolate using PoseNDF.'
    )
    parser.add_argument('--config', '-c', default='posendf/version2/config.yaml', type=str, help='Path to config file.')
    parser.add_argument('--ckpt_path', '-ckpt', default='posendf/version2/checkpoint_epoch_best.tar', type=str, help='Path to pretrained model.')
    parser.add_argument('--pose_file', '-pf', default='training_data/ACCAD/Female1General_c3d.npz', type=str, help='Path to noisy motion file')
    args = parser.parse_args()

    opt = load_config(args.config)

    main(opt, args.ckpt_path)