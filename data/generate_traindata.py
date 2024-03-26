import faiss
import numpy as np
import os
from pytorch3d.transforms import axis_angle_to_quaternion
import torch
import tyro

from data.create_data import PoseData, quat_flip
from data.data_splits import amass_splits
import data.dist_utils as dist_utils


def faiss_idx_np(amass_datasets, root_dir):
    all_pose = []

    for amass_dataset in amass_datasets:
        current_dataset_dir = os.path.join(root_dir,amass_dataset)
        seqs = sorted(os.listdir(current_dataset_dir))
        print(amass_dataset, len(seqs))


        for seq in seqs:
            if not 'npz' in seq:
                continue
            pose_seq = np.load(os.path.join(current_dataset_dir, seq))['pose_body'][:, :63]
            pose_seq = torch.from_numpy(pose_seq.reshape(len(pose_seq), 21, 3))
            pose_seq = axis_angle_to_quaternion(pose_seq).detach().numpy()
            pose_seq, _ = quat_flip(pose_seq)  

            all_pose.extend(pose_seq.reshape(len(pose_seq), 84))
    all_pose = np.array(all_pose)

    index = faiss.index_factory(84, "Flat")
    index.train(all_pose)
    index.add(all_pose)
    return  index, None, all_pose


def main(
        samples_on_manifold_path: str = './amass_samples', out_dir: str = './training_data',
        num_samples: int = 10000, runs: int = 1, metric: str = 'geo', batch_size: int = 1, device: str = 'cuda',
        k_faiss: int = 50, k_dist: int = 5, data_type: str = 'np'
        ):
    amass_datasets = sorted(amass_splits['train'])
    faiss_model, _, all_pose = faiss_idx_np(amass_datasets, samples_on_manifold_path)

    for amass_dataset in amass_datasets:
        current_dataset_dir = os.path.join(samples_on_manifold_path, amass_dataset)
        seqs = sorted(os.listdir(current_dataset_dir))
        for seq in seqs:
            current_dataset_out_dir = os.path.join(out_dir, amass_dataset)
            seq_out_dir = os.path.join(current_dataset_out_dir, seq)
            if  os.path.exists(seq_out_dir):
                print('Sample already processed....', amass_dataset,  seq)
                continue

            if not 'npz' in seq:
                continue

            os.makedirs(current_dataset_out_dir, exist_ok=True)

            #Create dataloader
            seq_sample_file = os.path.join(samples_on_manifold_path, amass_dataset, seq)
            query_data = PoseData(seq_sample_file, mode='query', batch_size=1, num_samples=num_samples, runs=runs)
            query_data_loader = query_data.get_loader()

            #create distance calcultor
            distance_calculator = getattr(dist_utils, metric)
            distance_calculator = distance_calculator(batch_size, device)

            batch_size = num_samples 
            k_faiss = k_faiss
            k_dist = k_dist

            all_dists = []
            all_poses = []
            all_nn_poses = []
            idx = 0
            for query_batch in query_data_loader:
                quer_pose_quat = query_batch.get('pose').to(device=device)[0]
                quer_pose_np = quer_pose_quat.reshape(-1, 84).detach().cpu().numpy()
            
                #for every query pose fine knn using faiss and then calculate exact enighbous using custom distance
                distances, neighbors = faiss_model.search(quer_pose_np, k_faiss)
                nn_poses = all_pose[neighbors].reshape(batch_size, k_faiss, 21, 4)

                if data_type == 'np':
                    nn_poses =  torch.from_numpy(nn_poses).to(device=device)

                print('calculating distance using geodesic distance calculator for {} poses'.format(len(quer_pose_quat)))
                dist, nn_id = distance_calculator.dist_calc(quer_pose_quat,nn_poses, k_faiss, k_dist)

                nn_id = nn_id.detach().cpu().numpy()
                nn_poses = nn_poses.detach().cpu().numpy()
                nn_pose = []
                for idx in range(batch_size):
                    nn_pose.append(nn_poses[idx][nn_id[idx]])
                
                all_dists.extend(dist.detach().cpu().numpy())
                all_poses.extend(quer_pose_quat.detach().cpu().numpy())
                all_nn_poses.extend(np.array(nn_pose))
                
            print('done for....{}, pose_shape...{}'.format(seq, len(all_poses)))
            np.savez(seq_out_dir, dist=np.array(all_dists), nn_pose=np.array(all_nn_poses), pose=np.array(all_poses))


if __name__ == "__main__":
    tyro.cli(main, description=__doc__)