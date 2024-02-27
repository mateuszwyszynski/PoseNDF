from pathlib import Path
import numpy as np
import tyro

def disturb_poses(motion_fpath: str, noise_magnitude: float = 0.01, out_dir: str = 'amass-disturbed'):
    """
    Disturb poses from a file by adding noise to the values on joints.
    """
    data = np.load(motion_fpath)
    poses = data['poses']
    random_noise = np.random.normal(0, noise_magnitude, poses[:, 3:66].shape)
    poses[:, 3:66] = poses[:, 3:66] + random_noise
    out_path = Path(out_dir, Path(*Path(motion_fpath).parts[1:]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path, trans=data['trans'], gender=data['gender'], mocap_framerate=data['mocap_framerate'],
        betas=data['betas'], dmpls=data['dmpls'], poses=poses
    )

if __name__ == "__main__":
    tyro.cli(disturb_poses)