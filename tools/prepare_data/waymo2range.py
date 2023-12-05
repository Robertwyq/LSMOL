import os
import math
import numpy as np
from PIL import Image
import argparse
import multiprocessing
import tensorflow.compat.v1 as tf

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2
tf.enable_eager_execution()

def read_root(root): 
    file_list = sorted(os.listdir(root))
    return file_list

def save_top_range(range_images, camera_projections,name):
    main_range = range_images[1][0]
    main_range_tensor = tf.reshape(tf.convert_to_tensor(value=main_range.data), main_range.shape.dims)
    main_camera  = camera_projections[1][0]
    main_camera_tensor = tf.reshape(tf.convert_to_tensor(value=main_camera.data), main_camera.shape.dims)
    np.savez(name,range_image = main_range_tensor.numpy()[:,:,0], camera_projections = main_camera_tensor.numpy())

def save_calibration(inc,path):
    H = 2650
    W = 64
    start = 180
    end = -180
    with open(path,"w") as f:
        f.write(str(H))
        f.write(';')
        f.write(str(W))
        f.write(';')
        f.write(str(start))
        f.write(';')
        f.write(str(end))
        for i in inc:
            f.write(';')
            f.write(str(math.degrees(i)))

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

def main(waymo_root,split,output_dir,token, process_num, debug, only_front):
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
        os.makedirs(output_dir,exist_ok=True)
        segment_img_dir = os.path.join(segment_dir,'image')
        os.makedirs(segment_img_dir,exist_ok=True)
        segmenr_range_dir = os.path.join(segment_dir,'range')
        os.makedirs(segmenr_range_dir,exist_ok=True)

        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
        frame_list = ["%06d" % (x) for x in range(199)]
        # jpg,png (img format)
        new_imgs = ['frame_'+t+'.jpg' for t in frame_list]
        i = 0
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # only extract front-view img 
            if only_front:
                for index, image in enumerate(frame.images):
                    img = tf.image.decode_jpeg(image.image)
                    img = np.array(img)
                    I = Image.fromarray(np.uint8(img))
                    save_path = os.path.join(segment_img_dir,new_imgs[i])
                    I.save(save_path)
                    break
            # extract all-view img
            else:
                for index, image in enumerate(frame.images):
                    img = tf.image.decode_jpeg(image.image)
                    img = np.array(img)
                    I = Image.fromarray(np.uint8(img))
                    frame_path = new_imgs[i].split('.')[0]+'_camera'+str(index)+'.jpg'
                    save_path = os.path.join(segment_img_dir,frame_path)
                    I.save(save_path)

            # extract range
            range_images, camera_projections,range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
            name = os.path.join(segmenr_range_dir,new_imgs[i].split('.')[0])
            save_top_range(range_images,camera_projections,name)

            # extract calibration
            inc = frame.context.laser_calibrations[-1].beam_inclinations
            path_name = os.path.join(segment_dir,'calibration.txt')
            save_calibration(inc,path_name)
            
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
    parser.add_argument('--debug', type=bool, default=False, help = 'only test for 5 segments')
    parser.add_argument('--only_front', default=False, action="store_true", help = 'only use front camera')
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)

    if args.process>1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.waymo_root,args.split,args.output_dir, token, args.process,args.debug, args.only_front))
        pool.close()
        pool.join()
    else:
        main(args.waymo_root,args.split,args.output_dir, 0, args.process, args.debug, args.only_front)
