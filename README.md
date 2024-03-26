


## Pose-NDF: Modeling Human Pose Manifolds with Neural Distance Fields
 
This repository contains official implementation of ECCV-2022 paper: Pose-NDF: Modeling Human Pose Manifolds with Neural Distance Fields ([Project Page](https://virtualhumans.mpi-inf.mpg.de/posendf/))

<p align="center">

  <img alt="org_scan" src="images/teaser.png" width="80%">
</p>

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

#### 2.2 Create training data

In this step random noise is added to poses that were obtained in the previous step.
This happens inside the `PoseData` class in the `data.create_data` script.
Noise is added only to a selected subset of indices: `indices = np.random.randint(0, len(quat_pose), num)`.
The noise is added with: `sampled_pose = sampled_pose + self.sigma[i]*np.random.rand(21,4)`

To generate data run:

    python -m data.generate_traindata

#### DEPRECATED 2.2 Create a script for generating training data:

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

#### DEPRECATED 2.3 Create  training data :
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

A pose is generated in two steps:

1. Assign random values to joints
2. Project the resulting pose onto the manifold

You can generate random plausible poses with:

    python -m experiments.sample_poses --config={} --ckpt-path={} --num-poses={} --poses-fpath={} --max-projection-distance={} --max-projection-steps={} --render --save-projection-steps

where:

 - `--config` (optional): the path to the config file for the trained PoseNDF model.
 Be sure to use the model and configuration files that match.
 Default is `'config.yaml'`.
 - `--ckpt-path` (optional): relative path (w.r.t. experiment root directory which is specified in the config file) to the checkpoint with a trained model,
 - `--num-poses` (optional): how many poses should be generated.
 Default is one.
 - `--poses-fpath` (optional): the path (relative to the place the script is executed) to an `npz` file containing poses with initial, random values assigned to joints.
 The poses should be represented with quaternions.
 Examples of such files can be found in the training data.
 If no file is provided, joint values for each pose are initialized randomly.
 - `--max-projection-dist` (optional): the maximum accepted distance to the manifold for the final poses.
 Default is 0.001.
 - `--max-projection-steps` (optional): if specified, this is the maximum number of projection steps.
 The projection algorithm will stop after specified number of iterations no matter what is the distance to the manifold for the resulting poses.
 Default is `None` in which case the algorithm does not stop until all poses are within the specified `--max-projection-dist`.
 - `--render` (optional): whether to render the initial random poses and the projected poses.
 If the flag is missing nothing is rendered.
 - `--save-projections-steps` (optional): whether to save an `.npz` file with poses obtained in each step of the projection algorithm.
 If the flag is missing nothing is saved.
        
### Pose interpolation
    python -m experiments.interpolation --config={} --ckpt-path={} --poses_fpath={} --num-steps={} --step-size={} --max-projection-distance={} --max-projection-steps={} --save-interpolation-steps

where:


 - `--config` (optional): the path to the config file for the trained PoseNDF model.
 Be sure to use the model and configuration files that match.
 Default is `'config.yaml'`.
 - `--ckpt-path` (optional): relative path (w.r.t. experiment root directory which is specified in the config file) to the checkpoint with a trained model,
 - `--poses-fpath` (optional): a path to an `npz` file containing poses (potentially with some noise).
 The poses should be represented with quaternions.
 Examples of such files can be found in the training data.
 The first and the last pose in the data are taken as the poses to interpolate between.
 If no file with poses is provided, joint values for both the start and the end pose are initialized randomly.
 - `--num-steps` (optional): how many interpolation steps will be performed
 - `--step-size` (optional): what step size should be used.
 Default is 0.1.
  - `--max-projection-dist` (optional): the maximum accepted distance to the manifold for the final poses.
 Default is 0.001.
 - `--max-projection-steps` (optional): if specified, this is the maximum number of projection steps.
 The projection algorithm will stop after specified number of iterations no matter what is the distance to the manifold for the resulting poses.
 Default is `None` in which case the algorithm does not stop until all poses are within the specified `--max-projection-dist`.
 - `--save-interpolation-steps` (optional): whether to save an `.npz` file with poses obtained in each step of the interpolation algorithm.
 If the flag is missing nothing is saved.

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

## Visualization

You can also install [`viser`](https://github.com/nerfstudio-project/viser) to run interactive visualizations.
I have prepared a script which is based on the example for SMPLX in the `viser` project.
It allows to do some visualization.
Note that this is still work in progress.

### Projection algorithm

If you have run the `experiments/sample_poses.py` script with an option to save projection steps

```python
def project(self, noisy_poses, iterations=100, save_projection_steps=True)
```

you should have created a file with consecutive poses generated by the projection algorithm.
Currently these are saved inside the current experiment directory inside the `projections_steps` folder.
With this file ready, you can run:

    python -m utils.trajecotry_visualization --model-path={} --poses-path={}
    python -m utils.trajectory_visualization --model-path={} --poses-path={} --config={} --checkpoint-path={}

where:

 - `--model-path` should be a path to the **`smplx`** model.
 Note that this should be the `smplx` not `smpl` model.
 In theory you can use also other models by specifying `--model-type` but for some reason `smpl` model does not work.
 More on that in [this issue](https://github.com/mateuszwyszynski/PoseNDF/issues/8)
 - `--poses-path` is the path to the `npz` file with saved poses.
 E.g. `'posendf/version2/projection_steps/9.npz'`
 - `--config` is the path to the configuration file for the PoseNDF model.
 E.g. `'posendf/replicate-version2/config.yaml'`
 - `--ckpt-path` (optional): relative path (w.r.t. experiment root directory which is specified in the config file) to the checkpoint with a trained model.

Open the link presented by the CLI in a browser.
You can play the animation in a loop by selecting `Playing`.
You can also control the pose index with a slider or the next / previous pose buttons.

Note that there is a read only field which shows the distance to the manifold for the current pose.

### AMASS raw

Similarly as in the paragraph above, you can visualize movement in the raw AMASS data.
You just have to specify a different `--poses-path`.


### Citation:
    @inproceedings{tiwari22posendf,
        title = {Pose-NDF: Modeling Human Pose Manifolds with Neural Distance Fields},
        author = {Tiwari, Garvita and Antic, Dimitrije and Lenssen, Jan Eric and Sarafianos, Nikolaos and Tung, Tony and Pons-Moll, Gerard},
        booktitle = {European Conference on Computer Vision ({ECCV})},
        month = {October},
        year = {2022},
        }

### Troubleshooting

 - if you get an import error when using `pytorch3d`
 (e.g. `libtorch_cuda_cu.so` cannot be found when calling `from pytorch3d import _C` or `from pytorch3d import renderer`)
 then I recommend to check if `pytorch3d` can be installed in a fresh environment.
 This might lead you to the cause of the problem.
 I have encountered such errors when running the code on GCP and managed to go around it by switching to a fresh VM with a more recent CUDA version.
 So I believe that in the end the problem was caused by some problems with installing `pytorch3d` with the CUDA setup from the other machine.
 - if you get an `ImportError: cannot import name 'bool' from 'numpy'` then you have to downgrade the version of numpy
 (this is mentioned in the installation steps).
 Discussed [here on Stack Overflow](https://stackoverflow.com/questions/74893742/how-to-solve-attributeerror-module-numpy-has-no-attribute-bool).
