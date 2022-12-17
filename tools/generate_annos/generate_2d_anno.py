import os
import numpy as np
import json
import argparse

WAYMO_CLASSES = ['unknown', 'object']


def main(root,box_name,min_objects,save_json_path):
    segs = sorted(os.listdir(root))

    global_id = 0
    object_id = 0
    images = []
    annotations = []
    categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]

    for seg in segs:
        bbox_path = os.path.join(root,seg,box_name)
        img_path = os.path.join(root,seg,'image')
        bboxs = sorted(os.listdir(bbox_path))
        nums = len(bboxs)
        for idx in range(nums):
            current_bbox_path = os.path.join(bbox_path,bboxs[idx])
            img_name = bboxs[idx].split('.')[0]+'.jpg'
            with open(current_bbox_path,'r') as load_f:
                bbox_data = json.load(load_f)
            if len(bbox_data)<=min_objects:
                continue
            current_img_path =os.path.join(img_path,img_name)
            images.append(dict(file_name=current_img_path, id=global_id, height=1280, width=1920))
            for key,value in bbox_data.items():
                bbox = value
                area = bbox[-1] * bbox[-2]
                annotations.append(dict(image_id=global_id,
                                                        bbox=bbox, area=area, category_id=1,
                                                        object_id=object_id,
                                                        tracking_difficulty_level=2,
                                                        detection_difficulty_level=2))
                object_id = object_id + 1                                
            global_id = global_id+1
        print('finish:',seg)
    
    with open(save_json_path,"w") as f:
        for i, anno in enumerate(annotations):
            anno['id'] = i #set as image frame ID
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str,default='/data/yuqi_wang/waymo_v1.2/waymo_lsmol',help='path to the lsmol root')
    parser.add_argument('--box_name',type=str,default='bbox_initial_sf_gt',help='box path name')
    parser.add_argument('--min_objects',type=int,default=1,help='minimum boxes per frame')
    parser.add_argument('--out_json_path',type=str,default='train_annotations_initial_sf_gt.json',help='json path name for 2d detection (coco format)')
    parser.add_argument('--outdir',type=str,default='/data/yuqi_wang/waymo_v1.2/lsmol',help='output dir for 2d annotations')
    args = parser.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    save_json_folder =  os.path.join(args.outdir,'pseudo_2d')
    os.makedirs(save_json_folder,exist_ok=True)
    save_json_path = os.path.join(save_json_folder,args.out_json_path)

    main(args.root,args.box_name,args.min_objects,save_json_path)