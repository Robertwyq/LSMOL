import numpy as np
import os
import argparse

def main(waymo_root):
    segs = sorted(os.listdir(waymo_root))

    for seg in segs:
        path = os.path.join(waymo_root,seg,'PC_ng')
        files = sorted(os.listdir(path))
        output_dir = os.path.join(waymo_root,seg,'point')
        os.makedirs(output_dir,exist_ok=True)
        first_frame = np.load(os.path.join(path,files[0]))
        initial_coord = first_frame['global_pc1'].mean(axis=0)

        for i in range(0,len(files)-1):
            tmp_path1 = os.path.join(path,files[i])
            tmp_path2 = os.path.join(path,files[i+1])
            global_p1 = np.load(tmp_path1)['global_pc1']
            global_p2 = np.load(tmp_path2)['global_pc1']
            camera_projections1 = np.load(tmp_path1)['cp_point']
            vehicle_p1 = np.load(tmp_path1)['point'][:,:3]
            global_gt = np.load(tmp_path1)['point'][:,3:]
            name = 'frame_'+ '%06d' % i + '.npz'
            global_p1 = global_p1-initial_coord
            global_p2 = global_p2-initial_coord
            # 10Hz (0.1s)
            if i==0:
                global_gt = global_gt*0.0
            else:
                global_gt = global_gt*0.1 
            cp1 = camera_projections1
            np.savez(os.path.join(output_dir,name),p1=global_p1,p2=global_p2,gt = global_gt,cp1 = cp1)
        print('finish:',seg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--waymo_root',default='/data/yuqi_wang/waymo_v1.2/waymo_lsmol',help='path to the waymo open dataset')
    args = parser.parse_args()

    main(args.waymo_root)