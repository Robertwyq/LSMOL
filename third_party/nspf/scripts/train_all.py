import os

waymo_lsmol_root = '/data/yuqi_wang/waymo_v1.2/waymo_lsmol'
segs = sorted(os.listdir(waymo_lsmol_root))
for i in range(len(segs)):
    name = segs[i]
    runfile = os.path.join(waymo_lsmol_root,name,'sceneflow_nsfp')
    if os.path.exists(runfile):
        print('skip')
        continue
    else:
        os.system("CUDA_VISIBLE_DEVICES={} bash scripts/train_single.sh {}".format(0,name))