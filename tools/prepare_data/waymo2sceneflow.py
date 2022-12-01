import os
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import multiprocessing

tf.enable_eager_execution()
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2

def read_root(root): 
    file_list = sorted(os.listdir(root))
    return file_list

def extract_flow(frame): 
    range_images_flow = {}
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_flow_compressed) > 0:
            flow_tensor = tf.io.decode_compressed(frame.lasers[0].ri_return1.range_image_flow_compressed, 'ZLIB')
            flow_tensor2 = tf.io.decode_compressed(frame.lasers[0].ri_return2.range_image_flow_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(flow_tensor.numpy()))
            ri2 = dataset_pb2.MatrixFloat()
            ri2.ParseFromString(bytearray(flow_tensor2.numpy()))
        range_images_flow[laser.name] = [ri,ri2]
    return range_images_flow

def extract_flow_on_camera(range_images_flow,camera_projections):
    main_range = range_images_flow[1][0]
    main_range_tensor = tf.reshape(tf.convert_to_tensor(value=main_range.data), main_range.shape.dims)
    main_camera  = camera_projections[1][0]
    main_camera_tensor = tf.reshape(tf.convert_to_tensor(value=main_camera.data), main_camera.shape.dims)
    return main_range_tensor.numpy(),main_camera_tensor.numpy()

def main(waymo_root,split,output_dir, token, process_num, debug):
    train_root = os.path.join(waymo_root,split)
    file_list = read_root(train_root)
    if debug:
        file_list = file_list[0:5]

    for s in range(len(file_list)):
        if s % process_num != token:
            continue
        filename = file_list[s]
        FILENAME = os.path.join(train_root,filename)
        sceneflow_dir = os.path.join(output_dir,filename.split('.')[0],'sceneflow_extra')
        os.makedirs(sceneflow_dir,exist_ok=True)
        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
        frame_list = ["%06d" % (x) for x in range(199)]
        new_imgs = ['frame_'+t+'.npy' for t in frame_list]
        i = 0
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            range_images, camera_projections,range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
            main_range = range_images[1][0]
            main_range_tensor = tf.reshape(tf.convert_to_tensor(value=main_range.data), main_range.shape.dims)
            main_range_array = main_range_tensor.numpy()
            # sceneflow
            range_images_flow = extract_flow(frame)
            flow_array, camera_array = extract_flow_on_camera(range_images_flow,camera_projections)
            front_mask = np.where((camera_array[:,:,0]==1)|(camera_array[:,:,3]==1),1,0)
            flow_mask = np.where((flow_array[:,:,3]>-1)&((np.abs(flow_array[:,:,0])>0) |(np.abs(flow_array[:,:,1])>0)  | (np.abs(flow_array[:,:,2])>0) ),1,0)
            point=[]
            instance_prop = np.where((front_mask[:,:]==1)&(flow_mask[:,:]==1),1,0)
            cam = camera_array[instance_prop==1]
            flo = flow_array[instance_prop==1]
            dep = main_range_array[instance_prop==1]
            for p in range(cam.shape[0]):
                if cam[p][0]==1:
                    point.append([cam[p][1],cam[p][2],flo[p][0],flo[p][1],flo[p][2],dep[p][0],dep[p][1]])
                elif cam[p][3]==1:
                    point.append([cam[p][4],cam[p][5],flo[p][0],flo[p][1],flo[p][2],dep[p][0],dep[p][1]])
            flow_name = os.path.join(sceneflow_dir,new_imgs[i])
            np.save(flow_name,np.array(point))
            # stop 
            i = i+1
            if i>198:
                break
        print('finish:',filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--waymo_root',default='/data/yuqi_wang/v1.2/scene-flow',help='path to the waymo open dataset')
    parser.add_argument('--split',default='train',choices=['train','valid'])
    parser.add_argument('--output_dir',default='/data1/yuqi_wang/waymo_lsmol',help='path to save the data')
    parser.add_argument('--process', type=int, default=1, help = 'num workers to use')
    parser.add_argument('--debug',type=bool,default=False)
    args = parser.parse_args()

    if args.process>1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.waymo_root,args.split,args.output_dir, token, args.process,args.debug))
        pool.close()
        pool.join()
    else:
        main(args.waymo_root,args.split,args.output_dir, 0, args.process, args.debug)
