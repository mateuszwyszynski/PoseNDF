from pathlib import Path
import numpy as np
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion
import torch
import tyro

from configs.config import load_config
from model.posendf import PoseNDF

def project_poses(
        poses_fpath: str, config: str = 'config.yaml', ckpt_path: str = 'checkpoint_epoch_best.tar',
        max_projection_dist: float = 0.001, max_projection_steps: int = None, out_dir: str = 'amass-projected-back'
        ):
    """
    Project poses from a file using a trained model.
    """
    opt = load_config(config)
    posendf = PoseNDF(opt)
    device= 'cuda:0'
    posendf.load_checkpoint_from_path(Path(posendf.experiment_dir, ckpt_path), device=device, training=False)
    data = np.load(poses_fpath)
    poses = torch.from_numpy(data['poses'].astype(np.float32)).to(device=device)
    joint_quaternions = axis_angle_to_quaternion(poses[:, 3:66].reshape(-1, 21, 3))
    projected_joint_quaternions = posendf.project(
        joint_quaternions,
        max_dist=max_projection_dist, max_steps=max_projection_steps
        )
    poses[:, 3:66] = quaternion_to_axis_angle(projected_joint_quaternions.detach().cpu()).reshape(-1, 63)
    out_path = Path(out_dir, Path(*Path(poses_fpath).parts[1:]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path, trans=data['trans'], gender=data['gender'], mocap_framerate=data['mocap_framerate'],
        betas=data['betas'], dmpls=data['dmpls'], poses=poses.cpu().detach().numpy()
    )

if __name__ == "__main__":
    tyro.cli(project_poses)
