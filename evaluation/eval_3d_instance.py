import torch
import pickle
import os
import cv2
import numpy as np
from PIL import Image
from mmdet3d.core.bbox import CameraInstance3DBoxes,get_box_type
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy import stats

def calculate_PR(class_tp, class_fp, class_score, class_gt_num):
    '''
    说明: 此函数用于计算某一类的PR曲线
    输入:
          class_tp:     list, 该类下的tp, 每个元素为0或1, 代表当前样本是否为正样本
          class_fp:     list, 该类下的fp, 每个元素为0或1, 代表当前样本是否为负样本
          class_score:  list, 该类下预测bbox对应的score
          class_gt_num: int,  类别数
    输出:
          P: list, 该类下的查准率曲线
          R: list, 该类下的查全率曲线
    '''

    # 按照score排序
    sort_inds = np.argsort(class_score)[::-1].tolist()
    tp = [class_tp[i] for i in sort_inds]
    fp = [class_fp[i] for i in sort_inds]
    # 累加
    tp = np.cumsum(tp).tolist()
    fp = np.cumsum(fp).tolist()
    # 计算PR
    P = [tp[i] / (tp[i] + fp[i]) for i in range(len(tp))]
    R = [tp[i] / class_gt_num for i in range(len(tp))]
    return P, R

def calculate_map_single(P, R):
    '''
    说明: 此函数用于计算PR曲线的面积, 即AP
    输入:
          P: list, 查准率曲线
          R: list, 查全率曲线
    输出:
          single_map: float, 曲线面积, 即AP
    '''
    mpre = np.concatenate(([0.], P, [0.]))
    mrec = np.concatenate(([0.], R, [1.]))
    for i in range(np.size(mpre) - 1, 0, -1):
        # mpre的平整化
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # 寻找mrec变化的坐标
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # 计算面积
    single_map = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return single_map

def compute_iou_one_frame(points_ins_id,pred_labels,gt_ins_id):
    pred_num = points_ins_id.shape[0]
    gt_num = np.where(gt_ins_id[:]!=-1,1,0).sum()
    match_num = np.where(pred_labels[:]!=-1,1,0).sum()
#     print('pred_num',pred_num)
#     print('gt_num',gt_num)
#     print('match_num',match_num)
    return pred_num,gt_num,match_num

def compute_ap_metric_one_frame(points_ins_id,pred_labels,pred_score,gt_ins_id,iou_thresh):
    ins_id = np.unique(points_ins_id)
    real_id = np.unique(gt_ins_id)
#     print('pred num:',len(ins_id))
#     print('gt_num:',len(real_id)-1)
    
    # 1.Match to GT
    ins_match = {}
    for iid in ins_id:
        obj = np.where(points_ins_id[:]==iid,1,0)
        real_labels = pred_labels[obj==1]
        most = stats.mode(real_labels)[0][0]
        ins_match[iid]=most
    count_num_list= []
    for y in ins_match.values():
        if y!=-1:
            count_num_list.append(y)
#     print('match num:',len(set(count_num_list)))
    
    # 2. GT
    gt_num = {}
    for rid in real_id:
        if rid!=-1:
            gt_p = np.where(gt_ins_id[:]==rid,1,0)
            gt_num[rid]=gt_p.sum()
    # 3. Pred
    ins_num = {}
    for j in ins_id:
        ins_num[j]=np.where(points_ins_id[:]==j,1,0).sum()
    
    # 4. Match 
    match_num={}
    for ins_key,ins_value in ins_match.items():
        if ins_value!=-1:
            m_obj = np.where((pred_labels[:]==ins_value) & (points_ins_id[:]==ins_key),1,0)
            match_num[ins_key]=m_obj.sum()
    
    IoU = {}
    for I_key,I_value in match_num.items():
        IoU[I_key]=match_num[I_key]/(gt_num[ins_match[I_key]]+ins_num[I_key]-match_num[I_key])
    
    final_result = {}
    fp = []
    tp = []
    fp_num = 0
    tp_num = 0
    fn_num = 0
    IoU_05 =[]
    scores = []
    for r_key,r_value in ins_match.items():
        if r_value==-1:
            fp_num=fp_num+1
            fp.append(1)
            tp.append(0)
        else:
            iou = IoU[r_key]
            if iou>=iou_thresh:
                tp_num = tp_num+1
                tp.append(1)
                fp.append(0)
                score_idx = np.where(points_ins_id[:]==r_key,1,0)
                score_mean = pred_score[score_idx==1].mean()
                IoU_05.append(iou)
                scores.append(score_mean)
    fn_num = len(real_id)-1-tp_num
            
    final_result['FP']=fp
    final_result['TP']=tp
    final_result['TP_NUM']=tp_num
    final_result['FP_NUM']=fp_num
    final_result['FN_NUM']=fn_num
    final_result['IoU_05']=IoU_05
    final_result['scores']=scores
    return final_result

def get_data_one_frame(idx,info,gt_ins,pred_ins):
    pred_test = pred_ins[idx]
    val_test = info[idx]
    gt_test = gt_ins[idx]
    pred_name = pred_test['img_metas'][0]['pts_filename'].split('/')[-1]
    val_name = val_test['point_cloud']['velodyne_path'].split('/')[-1]
    gt_name = gt_test['velodyne_path'].split('/')[-1]
    assert pred_name==val_name
    assert val_name==gt_name
    fg_ground = pred_test['points_fg']
    points_ins_id = pred_test['points_ins_id']
    rect = val_test['calib']['R0_rect'].astype(np.float32)
    Trv2c = val_test['calib']['Tr_velo_to_cam'].astype(np.float32)
    annos = val_test['annos']
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                          axis=1).astype(np.float32)
    box_type_3d, box_mode_3d = get_box_type('LiDAR')
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(box_mode_3d, np.linalg.inv(rect @ Trv2c))
    points = torch.tensor(fg_ground,dtype=torch.float32).cuda()
    pred_labels = gt_bboxes_3d.points_in_boxes(points).long()
    pred_labels = np.array(pred_labels.cpu())
    points_ins_id = np.array(points_ins_id)
    gt_ins_id = gt_test['labels']
    
    pred_score = np.array(pred_test['points_score'])
    return points_ins_id,pred_labels,pred_score,gt_ins_id

def compute_ap(info,gt_ins,pred_ins,iou_threshold=0.7):
    FP,TP,FN = 0,0,0
    pred,gt,match = 0,0,0
    FP_list,TP_list,scores_list = [],[],[]
    IoU_05 = []
    for idx in range(len(info)):
        points_ins_id,pred_labels,pred_score,gt_ins_id= get_data_one_frame(idx,info,gt_ins,pred_ins)
        # AP
        final_result = compute_ap_metric_one_frame(points_ins_id,pred_labels,pred_score,gt_ins_id,iou_thresh=iou_threshold)
        pred_num,gt_num,match_num = compute_iou_one_frame(points_ins_id,pred_labels,gt_ins_id)
        # IoU
        pred=pred+pred_num
        gt = gt+gt_num
        match = match+match_num
        
        FP=FP+final_result['FP_NUM']
        TP = TP+final_result['TP_NUM']
        FN = FN+final_result['FN_NUM']
        TP_list.extend(final_result['TP'])
        FP_list.extend(final_result['FP'])
        IoU_05.extend(final_result['IoU_05'])
        scores_list.extend(final_result['scores'])
        if idx%5000==0:
            print('finish:',idx)
    # Compute AP
    print('Compute AP:')
    P, R = calculate_PR(TP_list,FP_list,scores_list,TP+FN)
    single_ap = calculate_map_single(P,R)
    print('AP:',single_ap)
    # Compute IoU
    print('Compute IoU:')
    IoU = match/(pred+gt-match)
    print('IoU:',IoU)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    print('precision:',Precision)
    print('Recall:',Recall)


def main():
    # load basic information
    # generate from mmdetection3d (kitti_format/waymo_infos_val.pkl)
    pickle_data_path = '/waymo_root/kitti_format/waymo_infos_val.pkl'
    f0 = open(pickle_data_path,'rb')
    info = pickle.load(f0)

    # load gt instance segmentation
    # generate by generate_3d_instance_gt.py
    eval_gt_path = '/waymo_root/kitti_format/val_instance_gt.pkl'
    f1 = open(eval_gt_path,'rb')
    gt_ins = pickle.load(f1)

    # load pred instance segmentation
    # generate by the model
    pred_ins_path = '/log/results_eval_gt.pkl'
    f2 = open(pred_ins_path,'rb')
    pred_ins = pickle.load(f2)

    # change the threshold to get different AP results
    compute_ap(info,gt_ins,pred_ins,iou_threshold=0.7)

if __name__ == "__main__":
    main()