Please creae posendf environment using one of these:

##  1.Create environment using requirements.txt file
    conda create --name posendf --file requirements.txt


## 2. From scratch

### Create environment
    conda create -n posendf python=3.9
    conda activate posendf


### Install pytorch

    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


### Install other dependencies
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub
    conda install jupyter
    pip install scikit-image matplotlib imageio plotly opencv-python
    pip install trimesh
    pip install tensorboard
    pip install ipdb
    pip install smplx
    pip install chumpy
  
### Install pytorch3d and faiss-gpu
    conda install pytorch3d -c pytorch3d
    conda install -c pytorch faiss-gpu


Please install [pytorch3d](https://github.com/facebookresearch/pytorch3d) and [faiss](https://github.com/facebookresearch/faiss) using official documentation.

## 3. From scratch - NEW VERSION - in the following order works on GCP

TODO: convert this into a single `environment.yml`

### Create environment
    conda create -n posendf python=3.9
    conda activate posendf


### Install pytorch, pytorch3d and faiss-gpu

    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
    conda install pytorch3d -c pytorch3d
    conda install -c pytorch faiss-gpu

### installs needed for data preparation

    pip install smplx

### installs needed for training

    pip install tensorboard
    pip install scikit-learn

### installs needed for pose generation

    pip install opencv-python
    pip install chumpy
    conda install numpy=1.23.1
