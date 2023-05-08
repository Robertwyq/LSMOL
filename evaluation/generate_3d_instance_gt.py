import torch
import pickle
import os
import numpy as np
from PIL import Image
from mmdet3d.core.bbox import CameraInstance3DBoxes,get_box_type

def main():

    # change the path to your own path
    pickle_data_path = '/data/waymo_root/kitti_format/waymo_infos_val.pkl'
    root = '/data/waymo_root/kitti_format'

    f = open(pickle_data_path,'rb')
    info = pickle.load(f)

    num = len(info)
    print('total point clouds:',num)

    gt_instance_list = []

    for idx in range(num):
        gt_instance = {}
        info_c = info[idx]
        
        pc_path = os.path.join(root,info_c['point_cloud']['velodyne_path'])
        pc_data = np.fromfile(pc_path,dtype=np.float)
        pc_data = pc_data.reshape(-1,3) 

        rect = info_c['calib']['R0_rect'].astype(np.float32)
        Trv2c = info_c['calib']['Tr_velo_to_cam'].astype(np.float32)
        annos = info_c['annos']
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                            axis=1).astype(np.float32)
        box_type_3d, box_mode_3d = get_box_type('LiDAR')
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(box_mode_3d, np.linalg.inv(rect @ Trv2c))

        points = torch.tensor(pc_data,dtype=torch.float32).cuda()
        labels = gt_bboxes_3d.points_in_boxes(points).long()

        labels = np.array(labels.cpu())

        gt_instance['timestamp'] = info_c['timestamp']
        gt_instance['velodyne_path'] = info_c['point_cloud']['velodyne_path']
        gt_instance['labels'] = labels

        gt_instance_list.append(gt_instance)

        print('finish:',idx)

    print('saving for gt instance segmentation')

    pickle_out_path = os.path.join(root,'val_instance_gt.pkl')
    pickle_file = open(pickle_out_path,'wb')
    pickle.dump(gt_instance_list,pickle_file)
    pickle_file.close()


if __name__ == "__main__":
    main()