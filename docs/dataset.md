## Data Preparation
```bash
cd tools
# extract image,range_image,calibration (about 2 hours), we also support multi camera now
python ./prepare_data/waymo2range.py --only_front --process 24
# extract proposal from range image, generate proposal folder
cd initial_seg
python range2proposals.py  --process 32
# extract sceneflow and extra info, only front camera
python ./prepare_data/waymo2sceneflow.py --process 24
# extract point cloud (remove ground)
python ./prepare_data/waymo2point_noground --process 24
```
### Pipeline
```bash
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