import os
import numpy as np
import json
from sklearn import preprocessing
from scipy import stats
import hdbscan
import multiprocessing
import argparse

def generate_bbox(point_obj,enlarge_pixel = 2):
    x_min,x_max = max(0,int(point_obj[:,0].min())-enlarge_pixel),min(int(point_obj[:,0].max())+enlarge_pixel,1919)
    y_min,y_max = max(0,int(point_obj[:,1].min())-enlarge_pixel),min(int(point_obj[:,1].max())+enlarge_pixel,1279)
    w = x_max-x_min
    h = y_max-y_min
    return [x_min,y_min,w,h]

def filter_object_point_gtflow(proposal,camera_projections,sceneflow,low,high,v_t,cluster_size):
    front_mask = np.where((camera_projections[:,:,0]==1)|(camera_projections[:,:,3]==1),1,0)
    instance_mask = np.where(front_mask[:,:]==1,1,0)
    front_proposals = list(proposal[instance_mask==1])
    front_proposals_set = sorted(set(front_proposals))[1:]
    # print('after project to front:',len(front_proposals_set))
    filter_proposal = []
    for i in front_proposals_set:
        if front_proposals.count(i)>low and front_proposals.count(i)<high:
                filter_proposal.append(i)
    # print('after filter points:',len(filter_proposal))
    pre_color = {}
    mask = np.zeros((64,2650,3))
    for c in filter_proposal:
        rand_color = np.random.randint(1, 255, 3)
        pre_color[c]=rand_color
    instance_mask = np.zeros((64,2650))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if proposal[i,j] in filter_proposal:
                mask[i,j,:]=pre_color[proposal[i,j]]
                instance_mask[i,j] = proposal[i,j]
    point=[]
    for ins_id in filter_proposal:
        instance_prop = np.where((front_mask[:,:]==1)&(instance_mask==ins_id),1,0)
        cam = camera_projections[instance_prop==1]
        for p in range(cam.shape[0]):
            if cam[p][0]==1:
                point.append([cam[p][1],cam[p][2],ins_id])
            elif cam[p][3]==1:
                point.append([cam[p][4],cam[p][5],ins_id])

    motion_count = {}
    sceneflow_dict = {}

    if sceneflow.shape[0]>cluster_size:
        X = sceneflow[:,2:-1]
        X_scaled = preprocessing.scale(X)
        db = hdbscan.HDBSCAN(min_cluster_size=cluster_size,gen_min_span_tree=True).fit(X_scaled)
        labels = db.labels_
        movable_point = []
        for i in range(sceneflow.shape[0]):
            movable_point.append([sceneflow[i][0],sceneflow[i][1],labels[i]])
        for sf in range(len(movable_point)):
            pos = (int(movable_point[sf][0]),int(movable_point[sf][1]))
            v = movable_point[sf][2]
            sceneflow_dict[pos] = v
        ins_index = {}
        for ins_i in filter_proposal:
            v_label = []
            ins0 = [p for p in point if p[2]==ins_i]
            loc0 = [(p[0],p[1]) for p in point if p[2]==ins_i]
            count = 0
            for i in range(len(loc0)):
                if loc0[i] in sceneflow_dict.keys():
                    count+=1
                    v_label.append(sceneflow_dict[loc0[i]])
            rotio = count/len(loc0)
            motion_count[ins_i]=rotio
            if len(v_label)>0:
                most = stats.mode(v_label)[0][0]
                ins_index[ins_i]=most
        select_id = []
        for key,values in motion_count.items():
            if values>v_t:
                select_id.append(key)
        # print('after filter motion:',len(select_id))
        final_point = []
        new_id = []
        for s_id in select_id:
            new  = ins_index[s_id]+max(select_id)
            point_obj = [[p[0],p[1],new] for p in point if p[2]==s_id]
            final_point.extend(point_obj)
            new_id.append(new)
        new_id = set(new_id)
        # print('after merge:',len(new_id))
        return final_point,new_id
    else:
        for sf in range(sceneflow.shape[0]):
            pos = (int(sceneflow[sf][0]),int(sceneflow[sf][1]))
            v = [sceneflow[sf][2],sceneflow[sf][3],sceneflow[sf][4]]
            sceneflow_dict[pos] = v
        for ins_i in filter_proposal:
            loc0 = [(p[0],p[1]) for p in point if p[2]==ins_i]
            count = 0
            for i in range(len(loc0)):
                if loc0[i] in sceneflow_dict.keys():
                    count+=1
            rotio = count/len(loc0)
            motion_count[ins_i]=rotio
        select_id = []
        for key,values in motion_count.items():
            if values>v_t:
                select_id.append(key)
        return point,select_id


def main(waymo_root, token, process_num, high, low, v_t, cluster_size, output_name):

    segs = sorted(os.listdir(waymo_root))

    frame_list = ["%06d" % (x) for x in range(199)]
    new_json = ['frame_'+t+'.json' for t in frame_list]

    for s in range(len(segs)):
        seg = segs[s]
        if s % process_num != token:
            continue
        print('starting for info ',seg)

        range_root = os.path.join(waymo_root,seg,'range')
        ranges = sorted(os.listdir(range_root))
        sceneflow_root = os.path.join(waymo_root,seg,'sceneflow_extra')
        sceneflows = sorted(os.listdir(sceneflow_root))
        proposal_root = os.path.join(waymo_root,seg,'proposal')
        proposals = sorted(os.listdir(proposal_root))

        output_dir = os.path.join(waymo_root, seg, output_name)
        os.makedirs(output_dir,exist_ok=True)
        num = len(ranges)

        # gt sceneflow has no value for the first frame of each segment
        # so the index begin with 1
        for idx in range(1,num):
            range_path = os.path.join(range_root,ranges[idx])
            sceneflow_path = os.path.join(sceneflow_root,sceneflows[idx])
            proposal_path = os.path.join(proposal_root,proposals[idx])
            camera_projections = np.load(range_path)['camera_projections']
            sceneflow = np.load(sceneflow_path)
            proposal = np.load(proposal_path)
            point,final_ins = filter_object_point_gtflow(proposal,camera_projections,sceneflow,low,high,v_t,cluster_size)

            result = {}
            for s_id in final_ins:
                point_obj = [p for p in point if p[2]==s_id]
                bbox = generate_bbox(np.array(point_obj))
                result[int(s_id)] = bbox
            json_name = os.path.join(output_dir,new_json[idx])
            with open(json_name,"w") as dump_f:
                json.dump(result,dump_f)
        
        print('finish:',seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str,default='/data1/yuqi_wang/waymo_lsmol',help='path to the lsmol root')
    parser.add_argument('--process', type=int, default=1, help = 'num workers to use')
    parser.add_argument('--max_point',type = int, default=5000, help='parameter for filtering the proposals')
    parser.add_argument('--min_point',type = int, default=10, help='parameter for filtering the proposals')
    parser.add_argument('--motion_ratio',type = float, default=0.8, help='parameter for filtering the proposals')
    parser.add_argument('--cluster_size',type = int, default=15, help = 'hyper-parameter in hdbscan')
    parser.add_argument('--output_path',type=str, default='bbox_initial_sf_gt', help='path name save in the lsmol root')
    args = parser.parse_args()

    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.root, token, args.process, args.max_point,args.min_point,args.motion_ratio,args.cluster_size,args.output_path))
        pool.close()
        pool.join()
    else:
        main(args.root,0,args.process,args.max_point,args.min_point,args.motion_ratio,args.cluster_size,args.output_path)