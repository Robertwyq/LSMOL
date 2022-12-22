# <center>4D Unsupervised Object Discovery<center>
> [NeurIPS 2022 Spotlight] [**4D Unsupervised Object Discovery**](https://arxiv.org/pdf/2210.04801.pdf).
> Yuqi Wang, Yuntao Chen, [Zhaoxiang Zhang](https://zhaoxiangzhang.net)


## News
- **[2022/12/1]** https://zhuanlan.zhihu.com/p/588302740
- **[2022/10/10]** Paper released on <https://arxiv.org/pdf/2210.04801.pdf>
- **[2022/9/15]** 🔥*LSMOL* was accepted by NeurIPS 2022.


## Catalog
**The code is not available now, we will release it soon🚀**
- [] 3D Instance Segmentation
- [x] 2D Detection
- [x] NSPF for sceneflow estimation
- [x] 3D Instance Initialization Code
- [x] Data Processing Code 
- [x] Initialization


## Abstract
Object discovery is a core task in computer vision. While fast progresses have been made in supervised object detection, its unsupervised counterpart remains largely unexplored. With the growth of data volume, the expensive cost of annotations is the major limitation hindering further study.  Therefore, discovering objects without annotations has great significance. However, this task seems impractical on still-image or point cloud alone due to the lack of discriminative information. Previous studies underlook the crucial temporal information and constraints naturally behind multi-modal inputs. In this paper, we propose 4D unsupervised object discovery, jointly discovering objects from 4D data -- 3D point clouds and 2D RGB images with temporal information. We present the first practical approach for this task by proposing a ClusterNet on 3D point clouds, which is jointly iteratively optimized with a 2D localization network. Extensive experiments on the large-scale Waymo Open Dataset suggest that the localization network and ClusterNet achieve competitive performance on both class-agnostic 2D object detection and 3D instance segmentation, bridging the gap between unsupervised methods and full supervised ones.


## Methods
![method](figs/pipeline.png "model arch")


## Visualization
![vis](figs/prediction.png "prediction")


## Get Started


### Installation
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


### Data Preparation
```
cd tools
# extract image,range_image,calibration (about 2 hours)
python ./prepare_data/waymo2range.py --process 24
# extract sceneflow and extra info
python ./prepare_data/waymo2sceneflow.py --process 24
# extract point cloud (remove ground)
python ./prepare_data/waymo2point_noground --process 24
```
### Pipeline
```
# initial proposals by sceneflow
cd tools
python ./initial_seg/proposals2bbox_gt_flow.py --process 32
```
### Data Format
```
waymo_lsmol
├── segment-xxxx/
│   ├── image/
│   ├── range/
│   ├── proposal/
│   ├── PC_ng/
│   ├── sceneflow_extra/
│   ├── calibration.txt 
```

## Citation
Please consider citing our work as follows if it is helpful.
```
@article{wang20224d,
  title={4D Unsupervised Object Discovery},
  author={Wang, Yuqi and Chen, Yuntao and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2210.04801},
  year={2022}
}
```

## Acknowledgement 
Many thanks to the following open-source projects:
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
* [detectron2](https://github.com/facebookresearch/detectron2)  
* [depth_clustering](https://github.com/PRBonn/depth_clustering)
* [Neural_Scene_Flow_Prior](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior)
