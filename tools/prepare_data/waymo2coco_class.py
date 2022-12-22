import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import argparse
from pathlib import Path
import cv2
import json
from glob import glob
import os
import argparse

WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

def get_camera_labels(frame):
    if frame.camera_labels:
        return frame.camera_labels
    return frame.projected_lidar_labels

def extract_segment_frontcamera(tfrecord_files, out_dir, step):
    
    images = []
    annotations = []
    categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]
    image_globeid=0
    
    for segment_path in tfrecord_files:

        print(f'extracting {segment_path}')
        segment_path=Path(segment_path)
        segment_name = segment_path.name
        print(segment_name)
        segment_out_dir = out_dir / 'pseudo_anno'

        dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')
        
        for i, data in enumerate(dataset):
            if i % step != 0:
                continue

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            #get one frame

            context_name = frame.context.name
            frame_timestamp_micros = str(frame.timestamp_micros)

            for index, image in enumerate(frame.images):
                if image.name != 1: #Only use front camera
                    continue
                camera_name = open_dataset.CameraName.Name.Name(image.name)
                image_globeid = image_globeid + 1

                img = tf.image.decode_jpeg(image.image).numpy()
                image_name='_'.join([frame_timestamp_micros, camera_name]) #image name
                image_id = '/'.join([context_name, image_name]) #using "/" join, context_name is the folder
                #New: do not use sub-folder
                image_id = '_'.join([context_name, image_name])
                file_name = image_id + '.jpg'
                filepath = out_dir / file_name
                filepath.parent.mkdir(parents=True, exist_ok=True)

                images.append(dict(file_name=file_name, id=image_globeid, height=img.shape[0], width=img.shape[1], camera_name=camera_name))#new add camera_name
                print("current image id: ", image_globeid)
                # cv2.imwrite(str(filepath), img)

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
                            area = width * height
                            class_type = label.type
                            if class_type==3:
                                continue
                            elif class_type==4:
                                annotations.append(dict(image_id=image_globeid,
                                                    bbox=[x, y, width, height], area=area, category_id=3,
                                                    object_id=label.id,
                                                    tracking_difficulty_level=2 if label.tracking_difficulty_level == 2 else 1,
                                                    detection_difficulty_level=2 if label.detection_difficulty_level == 2 else 1))
                            else:
                                annotations.append(dict(image_id=image_globeid,
                                                    bbox=[x, y, width, height], area=area, category_id=label.type,
                                                    object_id=label.id,
                                                    tracking_difficulty_level=2 if label.tracking_difficulty_level == 2 else 1,
                                                    detection_difficulty_level=2 if label.detection_difficulty_level == 2 else 1))

    with (segment_out_dir / 'val_annotations_class.json').open('w') as f:
        for i, anno in enumerate(annotations):
            anno['id'] = i #set as image frame ID
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--waymo_root',default='/data/yuqi_wang/waymo_v1.2/waymo_format',help='path to the waymo open dataset')
    parser.add_argument('--split',default='valid',choices=['train','valid'])
    parser.add_argument('--out_dir',type=str,default='/data/yuqi_wang/waymo_v1.2/lsmol/val')
    parser.add_argument('--step',type=int, default=1, help = 'downsample rate: frame interval in a sequence')
    args = parser.parse_args()

    tfrecord_files = glob(os.path.join(args.waymo_root, args.split, "*.tfrecord")) 
    print(len(tfrecord_files))

    out_dir = Path(args.out_dir)
    extract_segment_frontcamera(tfrecord_files, out_dir, args.step)