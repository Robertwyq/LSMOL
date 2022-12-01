import os
import tensorflow.compat.v1 as tf
import numpy as np
import multiprocessing
import argparse
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2

def filter_pc(frame,proposal_mask,range_images,camera_projections,range_image_top_pose,range_images_flow):
    top_range = np.array(range_images[1][0].data).reshape(64,2650,-1)
    top_range[...,0] = top_range[...,0]*proposal_mask

    # add flow
    range_image_flow = range_images_flow[1][0]
    range_image_flow_tensor = tf.reshape(tf.convert_to_tensor(value=range_image_flow.data), range_image_flow.shape.dims)
    # add flow

    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    c = calibrations[0]
    points = []
    cp_points = []
    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(frame, range_images, range_image_top_pose, 0, False)
    range_image = range_images[c.name][0]
    range_image_tensor = tf.convert_to_tensor(top_range)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_cartesian = cartesian_range_images[c.name]
    # add flow
    result = tf.concat([range_image_cartesian,range_image_flow_tensor],axis=2)
    range_image_cartesian = result
    # add flow

    points_tensor = tf.gather_nd(range_image_cartesian,tf.compat.v1.where(range_image_mask))

    cp = camera_projections[c.name][0]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor,
                                    tf.compat.v1.where(range_image_mask))
    points.append(points_tensor.numpy())
    cp_points.append(cp_points_tensor.numpy())
    return cp_points[0],points[0]

def transform_global(points_all,frame):
    new_pcs = np.concatenate((points_all,np.ones(points_all.shape[0])[:,np.newaxis]),axis=1)
    T_1 = np.array(frame.pose.transform).reshape((4,4))
    global_pc1 = T_1 @ new_pcs.T
    global_pc1 = global_pc1.T[:,:3]
    return global_pc1

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

def main(waymo_root,split,output_dir,token, process_num, debug):
    train_root = os.path.join(waymo_root,split)
    file_list = read_root(train_root)
    if debug:
        file_list = file_list[0:5]

    for s in range(len(file_list)):
        if s % process_num != token:
            continue
        filename = file_list[s]
        FILENAME = os.path.join(train_root,filename)
        segment_dir = os.path.join(output_dir,filename.split('.')[0])
        segmenr_pc_dir = os.path.join(segment_dir,'PC_ng')
        os.makedirs(segmenr_pc_dir,exist_ok=True)

        proposal_root = os.path.join(output_dir,segment_dir,'proposal')
        proposals = sorted(os.listdir(proposal_root))

        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
        frame_list = ["%06d" % (x) for x in range(199)]
        new_file = ['frame_'+t+'.npz' for t in frame_list]
        i = 0

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            range_images, camera_projections,range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
            range_images_flow = extract_flow(frame)
            proposal_path = os.path.join(proposal_root,proposals[i])
            proposal = np.load(proposal_path)
            proposal_mask = np.where(proposal[:,:]!=0,1,0)
            cp_points_all,points_all = filter_pc(frame,proposal_mask,range_images,camera_projections,range_image_top_pose,range_images_flow)

            point_xyz=points_all[:,:3]

            global_pc1 = transform_global(point_xyz,frame)
            name = os.path.join(segmenr_pc_dir,new_file[i])
            np.savez(name,global_pc1=global_pc1,cp_point = cp_points_all,point = points_all)

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
