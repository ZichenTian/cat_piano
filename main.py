import os
import sys
import copy
import cv2

from video_capture import VideoFileVideoCapture
from detect_cat import CatDetector
from piano_key import PianoKey, KeyType, sum_up_all_key_img, fuse_img, init_piano

def out_of_image(img_shape, position):
    h = img_shape[0]
    w = img_shape[1]
    x = position[0]
    y = position[1]

    return x < 0 or x >= w or y < 0 or y >= h

def process_boxes(boxes):
    pass

def main(model_path, video_path):
    img_show_h = 1280
    img_show_w = 720
    basic_size = 100
    volume = 100
    rotate_img = False
    skip_frame = 1

    x_offset = int(basic_size * 0.5)
    y_offset = int(basic_size * 2)

    show_img_shape = (img_show_h, img_show_w, 3)

    # init cat detector
    cat_detector = CatDetector(model_path)
    detector_input_h, detector_input_w = cat_detector.model_runner.get_input_tensor_shape()[2:]
    scale_h = img_show_h / detector_input_h
    scale_w = img_show_w / detector_input_w

    # init video capture (from video file)
    cap = VideoFileVideoCapture(video_path)

    # init piano
    piano_keys = init_piano(img_show_h, img_show_w, basic_size, volume, x_offset, y_offset)

    # start to run
    cnt = 0
    if cap.initialize() == True:
        while True:
            cnt += 1
            # grab frame
            ret = cap.grab()
            if ret is False:
                break
            # skip frame
            if cnt % (skip_frame + 1) != 0:
                continue
            # decode frame
            ret, frame = cap.retrieve()
            if ret is False:
                break
            if rotate_img:
                frame = cv2.flip(cv2.transpose(frame), 1)     # rotate picture 90 degree
            show_img = cv2.resize(frame, (img_show_w, img_show_h))
            input_img = cv2.resize(show_img, (detector_input_w, detector_input_h))
            
            # detect cat
            boxes = cat_detector.run(input_img)

            # get box center
            center = (0, 0)
            if len(boxes) > 0:
                box = boxes[0]
                pt0 = (int(box[0] * scale_w), int(box[1] * scale_h))
                pt1 = (int(box[2] * scale_w), int(box[3] * scale_h))
                center = (int((pt0[0] + pt1[0]) / 2), int((pt0[1] + pt1[1]) / 2))
                cv2.rectangle(show_img, pt0, pt1, (0, 0, 255), 2)
                if out_of_image(show_img_shape, center):
                    center = (0, 0)
                cv2.circle(show_img, center, 15, (0, 0, 255), -1)
            
            # touch piano keys
            for key in piano_keys:
                key.play(center)

            # update & fuse piano img
            piano_img = sum_up_all_key_img(show_img_shape, piano_keys)
            show_img = fuse_img(show_img, piano_img, 0.5, 0.5)
            cv2.imshow('img', show_img)
            cv2.waitKey(10)
    else:
        raise Exception('init video capture failed')

if __name__ == '__main__':
    main('./best.onnx', 'video.mp4')
    