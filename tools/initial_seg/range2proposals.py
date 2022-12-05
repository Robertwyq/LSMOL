import sys
sys.path.append('../../third_party/depth_cluster/lib')
import segment
from PIL import Image
import numpy as np
import os
import argparse

def read_data(calib_path,range_path,img_path,imgs,ranges,idx):
    with open(calib_path, "r") as f: 
        data = f.read() 
    depth_path = os.path.join(range_path,ranges[idx])
    depth = np.load(depth_path)
    img_np = depth['range_image'].clip(min=0.)
    cp = depth['camera_projections']
    img = os.path.join(img_path,imgs[idx])
    img_new = Image.open(img)
    return data,img_np,cp,img_new


def main(root):

    segs = sorted(os.listdir(root))
    for seg_file in segs:
        # path
        output_path = os.path.join(root,seg_file,'proposal')
        os.makedirs(output_path,exist_ok=True)
        calib_path = os.path.join(root,seg_file,'calibration.txt')
        range_path = os.path.join(root,seg_file,'range')
        img_path = os.path.join(root,seg_file,'image')
        imgs = sorted(os.listdir(img_path))
        ranges = sorted(os.listdir(range_path))

        for idx in range(len(imgs)):
            data,img_np,cp,img_new = read_data(calib_path,range_path,img_path,imgs,ranges,idx)
            # generate proposal
            seg = segment.Segment(data)
            seg.extract_ground_and_instance(img_np)
            instance = seg.get_instances()
            name = imgs[idx].split('.')[0]+'.npy'
            pp_name = os.path.join(output_path,name)
            np.save(pp_name,instance)
        print('finish:',seg_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='/data1/yuqi_wang/waymo_lsmol',help='path for the data processed')
    parser.add_argument('--process', type=int, default=1, help = 'num workers to use')
    args = parser.parse_args()
    main(args.root)