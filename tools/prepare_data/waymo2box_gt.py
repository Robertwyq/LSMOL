import tensorflow.compat.v1 as tf
import matplotlib
matplotlib.use('Agg')
import multiprocessing
from waymo_open_dataset import dataset_pb2 as open_dataset
import argparse
from pathlib import Path
import cv2
import json
from glob import glob
import os

WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

def get_camera_labels(frame):
    if frame.camera_labels:
        return frame.camera_labels
    return frame.projected_lidar_labels

def extract_segment_frontcamera(segment_path, out_dir, step):
    
    categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]

    print(f'extracting {segment_path}')
    segment_path=Path(segment_path)
    segment_name = segment_path.name
    print(segment_name)
    name = segment_name.split('.')[0]
    segment_out_dir = os.path.join(out_dir,name,'box_gt')
    os.makedirs(segment_out_dir,exist_ok=True)

    frame_list = ["%06d" % (x) for x in range(200)]
    new_jsons = ['frame_'+t+'.json' for t in frame_list]

    dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')
    
    for i, data in enumerate(dataset):
        if i % step != 0:
            continue

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        #get one frame

        for index, image in enumerate(frame.images):
            if image.name != 1: #Only use front camera
                continue
            
            output_json_path = os.path.join(segment_out_dir,new_jsons[i])
            box_result = {}
            for camera_labels in get_camera_labels(frame):
                # Ignore camera labels that do not correspond to this camera.
                if camera_labels.name == image.name:
                    # Iterate over the individual labels.
                    for label in camera_labels.labels:
                        # object bounding box.
                        width = int(label.box.length)
                        height = int(label.box.width)
                        x = int(label.box.center_x - 0.5 * width)
                        y = int(label.box.center_y - 0.5 * height)
                        box_result[label.id] = [x, y, width, height]
            with open(output_json_path,"w") as dump_f:
                json.dump(box_result,dump_f)

    print('finish:',segment_name)

def main(waymo_root,split,output_dir, token, process_num, only_front, step, debug):
    tfrecord_files = sorted(glob(os.path.join(waymo_root, split, "*.tfrecord")))
    print(len(tfrecord_files))
    if debug:
        tfrecord_files = tfrecord_files[0:5]
    if only_front:
        for s in range(len(tfrecord_files)):
            if s % process_num != token:
                continue
            tfrecord_file = tfrecord_files[s]
            extract_segment_frontcamera(tfrecord_file, output_dir, step)
    else:
        for s in range(len(tfrecord_files)):
            if s % process_num != token:
                continue
            tfrecord_file = tfrecord_files[s]
            print('Not support yet!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--waymo_root',default='/data/yuqi_wang/v1.2/scene-flow',help='path to the waymo open dataset')
    parser.add_argument('--split',default='train',choices=['train','valid'])
    parser.add_argument('--output_dir',default='/data1/yuqi_wang/waymo_lsmol',help='path to save the data')
    parser.add_argument('--process', type=int, default=1, help = 'num workers to use')
    parser.add_argument('--only_front',type=bool,default=True,help = 'only use the front camera')
    parser.add_argument('--step',type=int, default=1, help = 'downsample rate: frame interval in a sequence')
    parser.add_argument('--debug',type=bool,default=False)
    args = parser.parse_args()

    if args.process>1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.waymo_root,args.split,args.output_dir, token, args.process,args.only_front, args.step, args.debug))
        pool.close()
        pool.join()
    else:
        main(args.waymo_root,args.split,args.output_dir, 0,args.process, args.only_front, args.step, args.debug)

