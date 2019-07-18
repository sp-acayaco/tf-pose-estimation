import json
import os

import argparse
import logging
import time

import cv2
import numpy as np
import ffmpeg

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    
    tags = meta_dict['streams'][0]['tags']
    rotation = tags.get('rotate')
    
    if rotation:
        if int(rotation) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(rotation) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(rotation) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
            
    return rotateCode
    
def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode)


def get_centroid(human):
    x = []
    y = []
    for i in range(common.CocoPart.Background.value):
        if i not in human.body_parts.keys():
            continue
        body_part = human.body_parts[i]
        x.append(body_part.x)
        y.append(body_part.y)
    x = sum(x)/len(x)
    y = sum(y)/len(y)
    return x, y


def human2dict(human):
    out = {}
    for i in range(common.CocoPart.Background.value):
        if i not in human.body_parts.keys():
            continue
        body_part = human.body_parts[i]
        out[i] = {'x': body_part.x, 'y': body_part.y}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='1312x736', help='network input resolution. default=1312x736')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    
    cap = cv2.VideoCapture(args.video)
    
    # check if video requires rotation
    rotateCode = check_rotation(args.video)

    state = {}

    if not cap.isOpened():
        print("Error opening video stream or file")
    else:
        # First frame
        ret_val, image = cap.read()
        # check if the frame needs to be rotated
        if rotateCode is not None:
            image = correct_rotation(image, rotateCode)

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        if not args.showBG:
            image = np.zeros(image.shape)

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        def select_target(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                image_h, image_w = image.shape[:2]
                pos_x = x / image_w
                pos_y = y / image_h
                param['target_centroid'] = (pos_x, pos_y)


        win_name = 'tf-pose-estimation result'
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, select_target, state)
        cv2.imshow(win_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # get centroid of target human in first frame
    target_centroid = state['target_centroid']

    timeseries_data = []

    while cap.isOpened():
    
        ret_val, image = cap.read()
        
        # check if the frame needs to be rotated
        if rotateCode is not None:
            image = correct_rotation(image, rotateCode)

        if not args.showBG:
            image = np.zeros(image.shape)

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        for human in humans:
            c = get_centroid(human)
            if target_centroid[0] - 0.05 < c[0] < target_centroid[0] + 0.05 and \
               target_centroid[1] - 0.05 < c[1] < target_centroid[1] + 0.05:
                target_centroid = c
                break

        timeseries_data.append(human2dict(human))

        image = TfPoseEstimator.draw_humans(image, [human], imgcopy=False)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        fps_time = time.time()
        cv2.imshow('tf-pose-estimation result', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

    path = os.path.basename(args.video) + '.json'
    with open(path, 'w') as f:
        json.dump(timeseries_data, f)

logger.debug('finished+')
