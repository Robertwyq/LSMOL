## Installation
- a. Create a conda virtual environment and activate it.
```
conda create -n lsmol python=3.7 -y
conda activate lsmol

# pytorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# waymo v1.2 
pip install waymo-open-dataset-tf-2-4-0==1.4.1 

# NSFP
conda install -c open3d-admin open3d==0.9.0
conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

pip install torch-geometric==1.6.3
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
pip install shapely
pip install mayavi
pip install PyQt5

# Depth Cluster
sudo apt-get install libeigen3-dev
sudo apt-get install libgtest-dev
sudo apt-get install libboost-all-dev
sudo apt-get install python-numpy

mkdir build
cd build
cmake ..
make -j4

# CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# HDBSCAN 
pip install hdbscan

# 3D Tracker
git clone https://github.com/ImmortalTracker/ImmortalTracker 

# Detectron2
cd third_party
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```