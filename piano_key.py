import os
import sys
import cv2
import copy
import numpy as np

from enum import Enum
from mingus.midi import fluidsynth
fluidsynth.init('/usr/share/sounds/sf2/FluidR3_GM.sf2', 'alsa')

class KeyType(Enum):
    WHITE_KEY = 0, 
    WHITE_KEY_LEFT = 1, 
    WHITE_KEY_RIGHT = 2, 
    BLACK_KEY = 3

class PianoKey(object):
    def __init__(self, key_type, note, anchor, volume):
        self.key_type = key_type
        self.note = note
        self.anchor = anchor
        self.volume = volume
        self.pressed = False
    
    def play_note(self):
        fluidsynth.play_Note(self.note, 0, self.volume)
    
    def is_hitted(self, position):
        x = position[0]
        y = position[1]
        if self.graph_layer[y, x, 0] > 0:
            return True
        return False
    
    def play(self, position):
        if self.is_hitted(position):
            if self.pressed == False:
                self.play_note()
            self.pressed = True
        else:
            self.pressed = False
    
    def show_graph_layer(self):
        if not self.pressed:
            return self.graph_layer
        
        if self.key_type == KeyType.BLACK_KEY:
            temp_graph_layer = copy.deepcopy(self.graph_layer)
            temp_graph_layer[temp_graph_layer == 10] = 100
            return temp_graph_layer
        else:
            temp_graph_layer = copy.deepcopy(self.graph_layer)
            temp_graph_layer[:,:,0] = 0
            return temp_graph_layer

    def init_graph_layer(self, img_size, basic_size):
        self.graph_layer = np.zeros(img_size, dtype=np.uint8)
        anchor_x = self.anchor[0]
        anchor_y = self.anchor[1]

        black_key_width = int(basic_size * 0.5)
        black_key_height = int(basic_size * 3)
        white_key_width = int(basic_size * 1)
        white_key_height = int(basic_size * 5)

        if self.key_type == KeyType.WHITE_KEY:
            # upper part rectangle
            left = anchor_x - int((white_key_width - black_key_width) / 2)
            right = anchor_x + int((white_key_width - black_key_width) / 2)
            up = anchor_y
            down = anchor_y + black_key_height

            cv2.rectangle(self.graph_layer, (left, up), (right, down), (255, 255, 255), -1)
            cv2.rectangle(self.graph_layer, (left, up), (right, down), (5, 5, 5), 2)
            
            # lower part rectangle
            left = anchor_x - int(white_key_width / 2)
            right = anchor_x + int(white_key_width / 2)
            up = down
            down = anchor_y + white_key_height

            cv2.rectangle(self.graph_layer, (left, up), (right, down), (255, 255, 255), -1)
            cv2.rectangle(self.graph_layer, (left, up), (right, down), (5, 5, 5), 2)

        elif self.key_type == KeyType.WHITE_KEY_LEFT:
            # upper part rectangle
            left = anchor_x - int(white_key_width / 2)
            right = anchor_x + int((white_key_width - black_key_width) / 2)
            up = anchor_y
            down = anchor_y + black_key_height

            cv2.rectangle(self.graph_layer, (left, up), (right, down), (255, 255, 255), -1)
            cv2.rectangle(self.graph_layer, (left, up), (right, down), (5, 5, 5), 2)

            # lower part rectangle
            left = anchor_x - int(white_key_width / 2)
            right = anchor_x + int(white_key_width / 2)
            up = down
            down = anchor_y + white_key_height

            cv2.rectangle(self.graph_layer, (left, up), (right, down), (255, 255, 255), -1)
            cv2.rectangle(self.graph_layer, (left, up), (right, down), (5, 5, 5), 2)
        
        elif self.key_type == KeyType.WHITE_KEY_RIGHT:
            # upper part rectangle
            left = anchor_x - int((white_key_width - black_key_width) / 2)
            right = anchor_x + int(white_key_width / 2)
            up = anchor_y
            down = anchor_y + black_key_height

            cv2.rectangle(self.graph_layer, (left, up), (right, down), (255, 255, 255), -1)
            cv2.rectangle(self.graph_layer, (left, up), (right, down), (5, 5, 5), 2)
            
            # lower part rectangle
            left = anchor_x - int(white_key_width / 2)
            right = anchor_x + int(white_key_width / 2)
            up = down
            down = anchor_y + white_key_height

            cv2.rectangle(self.graph_layer, (left, up), (right, down), (255, 255, 255), -1)
            cv2.rectangle(self.graph_layer, (left, up), (right, down), (5, 5, 5), 2)
        
        elif self.key_type == KeyType.BLACK_KEY:
            left = anchor_x - int(black_key_width / 2)
            right = anchor_x + int(black_key_width / 2)
            up = anchor_y
            down = anchor_y + black_key_height

            cv2.rectangle(self.graph_layer, (left, up), (right, down), (10, 10, 10), -1)

def sum_up_all_key_img(img_size, piano_keys):
    key_img = np.zeros(img_size, dtype=np.uint8)
    for key in piano_keys:
        key_img += key.show_graph_layer()
    return key_img

def fuse_img(img1, img2, alpha, beta):
    img = (img1 * alpha).astype(np.uint8)
    img += (img2 * beta).astype(np.uint8)
    return img

if __name__ == '__main__':
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    img[:,:] = (255, 255, 0)
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
        key.init_graph_layer(img.shape, basic_size)
    
    y = 200
    for x in range(0, 600, 5):
        position = (x, y)
        for key in piano_keys:
            key.play(position)
        piano_img = sum_up_all_key_img(img.shape, piano_keys)
        cv2.imshow('img_update', fuse_img(img, piano_img, 0.5, 0.5))
        cv2.waitKey(10)




