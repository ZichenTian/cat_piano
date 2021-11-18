import os
import sys
import copy
import cv2

from video_capture import VideoFileVideoCapture
from detect_cat import CatDetector
from piano_key import PianoKey, KeyType, sum_up_all_key_img, fuse_img

def out_of_image(img_shape, position):
    h = img_shape[0]
    w = img_shape[1]
    x = position[0]
    y = position[1]

    return x < 0 or x >= w or y < 0 or y >= h

def main(model_path, video_path):
    cat_detector = CatDetector(model_path)
    cap = VideoFileVideoCapture(video_path)

    img_shape = (640, 640, 3)

    basic_size = 80
    x_offset = int(basic_size * 0.75)
    y_offset = int(basic_size * 0.75)
    volume = 100

    piano_keys = [
        PianoKey(KeyType.WHITE_KEY_LEFT,    55, (x_offset + basic_size * 0, y_offset),   volume), 
        PianoKey(KeyType.WHITE_KEY,         57, (x_offset + basic_size * 1, y_offset),   volume), 
        PianoKey(KeyType.WHITE_KEY_RIGHT,   59, (x_offset + basic_size * 2, y_offset),   volume), 
        PianoKey(KeyType.WHITE_KEY_LEFT,    60, (x_offset + basic_size * 3, y_offset),   volume), 
        PianoKey(KeyType.WHITE_KEY,         62, (x_offset + basic_size * 4, y_offset),   volume), 
        PianoKey(KeyType.WHITE_KEY,         64, (x_offset + basic_size * 5, y_offset),   volume), 
        PianoKey(KeyType.WHITE_KEY_RIGHT,   66, (x_offset + basic_size * 6, y_offset),   volume), 

        PianoKey(KeyType.BLACK_KEY,         56, (x_offset + int(basic_size * 0.5), y_offset),   volume), 
        PianoKey(KeyType.BLACK_KEY,         58, (x_offset + int(basic_size * 1.5), y_offset),   volume), 
        PianoKey(KeyType.BLACK_KEY,         61, (x_offset + int(basic_size * 3.5), y_offset),   volume), 
        PianoKey(KeyType.BLACK_KEY,         63, (x_offset + int(basic_size * 4.5), y_offset),   volume), 
        PianoKey(KeyType.BLACK_KEY,         65, (x_offset + int(basic_size * 5.5), y_offset),   volume), 
    ]

    for key in piano_keys:
        key.init_graph_layer(img_shape, basic_size)

    if cap.initialize() == True:
        while True:
            ret, frame = cap.getFrame()
            frame = cv2.resize(frame, (640, 640))
            # frame = cv2.flip(cv2.transpose(frame), 1)
            box = cat_detector.run(copy.deepcopy(frame))
            center = (0, 0)
            if len(box) > 0:
                pt0 = (int(box[0][0]), int(box[0][1]))
                pt1 = (int(box[0][2]), int(box[0][3]))
                cv2.rectangle(frame, pt0, pt1, (0,0,255), 2)
                center = (int((pt0[0] + pt1[0]) / 2), int((pt0[1] + pt1[1]) / 2))
                if out_of_image(img_shape, center):
                    center = (0, 0)
                cv2.circle(frame, center, 5, (0,0,255), -1)
            for key in piano_keys:
                key.play(center)
            piano_img = sum_up_all_key_img(img_shape, piano_keys)
            frame = fuse_img(frame, piano_img, 0.5, 0.5)
            cv2.imshow('img', frame)
            cv2.waitKey(10)
    else:
        raise Exception('init video capture failed')

if __name__ == '__main__':
    main('./catDetectorOp11.onnx', 'video.mp4')
    