


## Pose-NDF: Modeling Human Pose Manifolds with Neural Distance Fields
 
This repository contains official implementation of ECCV-2022 paper: Pose-NDF: Modeling Human Pose Manifolds with Neural Distance Fields ([Project Page](https://virtualhumans.mpi-inf.mpg.de/posendf/))

<p align="center">

  <img alt="org_scan" src="images/teaser.png" width="80%">
</p>

## UPDATE:
Please use branch [version2](https://github.com/garvita-tiwari/PoseNDF/tree/version2) 

## Installation: 
Please follow [INSTALL.md](INSTALL.md)


## Training and Dataset

#### 1. Download AMASS: Store in a folder "amass_raw"". You can train the model for SMPL/SMPL+H or SMPL+X. 
    https://amass.is.tue.mpg.de/

One has to download the SMPL model from: https://smpl.is.tue.mpg.de/.
Currently I am using "version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)"


#### 2.1 Sample poses from AMASS:
This is the data preparation step based on
[VPoser data preparation](https://github.com/nghorbani/human_body_prior/tree/master/src/human_body_prior/data).
If you already have the data processed, you can skip this step.

    python -m data.sample_poses

By default, the raw AMASS data is assumed to be in the `./amass_raw` directory and the output is stored in `./amass_samples`.
One can change this behavious by providing additional arguments. Use:

 - `--amass_dir <amass_raw_dir>` to specify directory with raw AMASS data
 - `--sampled_pose_dir <output_dir>` to specify output directory

If you would like to use only a subset of data from AMASS, you should change the predefined variable `amass_splits` in the `data/data_splits.py` script.

TODO:
Why the following sentence was here in the original repo?
 - You just need to convert .pt file to .npz file.

#### 2.2 Create a script for generating training data:

In this step a bash script `train_data.sh` for training data generation is created in the project root directory.

    python -m data.prepare_data

By default, the input data is assumed to be in the `./amass_samples` directory (default value from the previous step) and the generated training data will be stored in `./training_data`.

One can change the default behaviour by providing additional arguments:

 - `--sampled_data <sampled_amass_poses_dir>` to specify the directory with the samples generated in the previous step
 - `--out_path <path_for_training_data>` to specify the directory were the training data should be stored 
 - `--bash_file <bash_file_name>` to use a different name for the generated bash script 

TODO:
Deal with these instructions about using slurm:
 - If you are using slurm then add "--use_slurm" and change the path on environment and machine specs in L24:L30 in data/prepare_data.py

#### 2.3 Create  training data :
Run the bash script (if needed change to your shell):

    bash train_data.sh

TODO:
Clarify these instructions:
 - During training the dataloader reads file form data_dir/. You can now delete the amass_raw directory. 
 - For all our experiments, we use the same settings as used in VPoser data preparation step.

#### 3. Edit configs/<>.yaml for different experimental setup
    experiment:
        root_dir: directory for training data/models and results
    model:     #Network acrhitecture
        ......
    training:  #Training parameters
        ......
    data:       #Training sample details
        .......

Root directory will contain dataset, trained models and results. 

#### 4. Training Pose-NDF :
    python trainer.py --config=configs/amass.yaml

amass.yaml contains the configs used for the pretrained model. 

#### 4. Download pre-trained model :  [Pretrained model](https://nextcloud.mpi-klsb.mpg.de/index.php/s/4zxN93WL769pSAK) 

Latest model: version2/
You can also find the corresponding config file in the same folder


## Inference 

Pose-NDF is a continuous model for plausible human poses based on neural distance fields (NDFs). This can be used to project non-manifold points on the learned manifold and hence act as prior for downstream tasks.


### Pose generation

    python -m experiments.sample_poses --config={} --ckpt_path={} --noisy_pose={} --outpath_folder={}

where:

 - `config`: path to the config file for the model.
 Be sure to use the model and configuration file that match,
 - `ckpt_path`: path to the checkpoint with a trained model,
 - `noisy_pose`: is a path to an `npz` file containing random poses in quaternions,
Examples of such files can be found in the training data.
 - `outpath_folder`: where to save rendered initial and projected pose images.

outpath_folder: save rendered initial and projected pose.
        

### Pose interpolation
    python -m experiments.interpolation --config={} --ckpt_path={} --pose_file={}

where parameters are the same as in _Pose generation_ with `pose_file` parameter being the equivalent of `noisy_pose` (no idea why...)

The `ipdb` package is used by this script.
Basically:
 - type `n` and press `Enter` to execute the next line
 - or type `c` and press Enter to execute until the next breaking point

More info how to use IPDB can be found [here](https://www.geeksforgeeks.org/using-ipdb-to-debug-python-code/) or [here](https://wangchuan.github.io/coding/2017/07/12/ipdb-cheat-sheet.html)

TODO:
1. Most likely we do not want to rely on IPDB.
Not intuitive and badly documented.
2. More importantly, I believe there is no code which actually performs pose interpolation.
The `experiments/interpolation.py` script simply generates two poses and that's all.
I'm pretty sure that nothing else happens, but I will confirm when I have a really good understanding of the code.

### Motion denoising

TODO: This section was not revised, because I do not have the noisy data.
Have to figure this out.

     python experiment/motion_denoise.py --config=configs/amass.yaml  --motion_data=<motion data folder> --ckpt_path={}  --outpath_folder={} --bm_dir_path={}

  
Motion data file is .npz file which contains "body_pose", "betas", "root_orient". This is generated using: https://github.com/davrempe/humor/tree/main/humor/datasets
bm_dir_path: path to SMPL body model

### Image based 3d pose estimation

TODO: Didn't cover this section, because it is not of the main interest for us currently.

     1. Run openpose to generate 2d keypoints for given image(https://github.com/CMU-Perceptual-Computing-Lab/openpose).
     2. python experiment/image_fitting.py --config=configs/amass.yaml  --image_dir=<image data dir>


Both image and corresponding keypoint should be in same directory with <image_name>.jpg and <image_name>.json being the image and 2d keypoints file respectively.


### Citation:
    @inproceedings{tiwari22posendf,
        title = {Pose-NDF: Modeling Human Pose Manifolds with Neural Distance Fields},
        author = {Tiwari, Garvita and Antic, Dimitrije and Lenssen, Jan Eric and Sarafianos, Nikolaos and Tung, Tony and Pons-Moll, Gerard},
        booktitle = {European Conference on Computer Vision ({ECCV})},
        month = {October},
        year = {2022},
        }

