import numpy as np
from pathlib import Path
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle, standardize_quaternion
import torch
import tyro


def disturb_poses(poses_fpath: str, noise_magnitude: float = 0.01, out_dir: str = 'amass-disturbed'):
    """
    Disturb poses from a file by adding noise to the values on joints.
    """
    data = np.load(poses_fpath)
    poses = data['poses']

    body_angles = torch.from_numpy(poses[:, 3:66].reshape(-1, 21, 3))
    body_quats = standardize_quaternion(axis_angle_to_quaternion(body_angles))

    body_noise_quats = body_quats + torch.normal(0, noise_magnitude, body_quats.shape)
    body_noise_quats = standardize_quaternion(torch.nn.functional.normalize(body_noise_quats,dim=-1))

    body_noise_angles = quaternion_to_axis_angle(body_noise_quats).detach().numpy()
    poses[:, 3:66] = body_noise_angles.reshape(-1, 63)
    out_path = Path(out_dir, Path(*Path(poses_fpath).parts[1:]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path, trans=data['trans'], gender=data['gender'], mocap_framerate=data['mocap_framerate'],
        betas=data['betas'], dmpls=data['dmpls'], poses=poses
    )


if __name__ == "__main__":
    tyro.cli(disturb_poses)