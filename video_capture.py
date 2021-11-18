import os
import sys
import cv2
import copy
import time
import threading

class VideoCaptureBase(object):
    def __init__(self):
        self.type = "VideoCaptureBase"
    
    def initialize(self):
        return True
    
    def release(self):
        return True
    
    def getFrame(self):
        return None

class VideoFileVideoCapture(object):
    def __init__(self, file_path):
        self.type = "VideoFileVideoCapture"
        self.file_path = file_path
        self.cap = cv2.VideoCapture()
    
    def initialize(self):
        if self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            print("initialize " + self.type + " failed: open video capture failed.")
            self.cap.release()
            return False
        
        ret, frame = self.cap.read()
        if ret == False or frame is None:
            print("initialize " + self.type + " failed: read frame failed.")
            self.cap.release()
            return False

        return True
    
    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        return True
    
    def getFrame(self):
        return self.cap.read()

class RTSPVideoCaptureBasic(VideoCaptureBase):
    def __init__(self, rtsp_addr):
        super().__init__()
        self.type = "RTSPVideoCaptureBasic"
        self.rtsp_addr = rtsp_addr
        self.cap = cv2.VideoCapture()
    
    def initialize(self):
        if self.cap.isOpened():
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.rtsp_addr)
        if not self.cap.isOpened():
            print("initialize " + self.type + " failed: open video capture failed.")
            self.cap.release()
            return False
        
        ret, frame = self.cap.read()
        if ret == False or frame is None:
            print("initialize " + self.type + " failed: read frame failed.")
            self.cap.release()
            return False
        
        return True
    
    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        return True
    
    def getFrame(self):
        return self.cap.read()
    
    def grab(self):
        return self.cap.grab()
    
    def retrieve(self):
        return self.cap.retrieve()

class RTSPVideoCaptureAsync(VideoCaptureBase):
    def __init__(self, rtsp_addr):
        super().__init__()
        self.type = "RTSPVideoCaptureAsync"
        self.rtsp_addr = rtsp_addr
        self.basic_video_capture = RTSPVideoCaptureBasic(self.rtsp_addr)
        self.mutex = threading.Lock()
        self.thread = None
        self.frame = None
        self.finish = False
    
    def initialize(self):
        if self.basic_video_capture.initialize() == False:
            print("initialize " + self.type + " failed: init basic video capture failed.")
            return False
        
        ret, self.frame = self.basic_video_capture.getFrame()

        self.thread = threading.Thread(target=self.run, args=())
        if self.thread is None:
            print("initialize " + self.type + " failed: create thread failed.")
            self.basic_video_capture.release()
            return False

        self.finish = False
        self.thread.start()
        return True
    
    def release(self):
        self.finish = True
        self.thread.join()
        return self.basic_video_capture.release()
    
    def run(self):
        print("thread create")
        while self.finish == False:
            time.sleep(0.01)
            self.mutex.acquire()
            _, self.frame = self.basic_video_capture.getFrame()
            self.mutex.release()
    
    def getFrame(self):
        self.mutex.acquire()
        frame = copy.deepcopy(self.frame)
        self.mutex.release()
        return True, frame

if __name__ == "__main__":
    # cap = RTSPVideoCaptureAsync("rtsp://admin:abcd1234@192.168.1.64:554//Streaming/Channels/1")
    cap = VideoFileVideoCapture("./VID_20211115_232518.mp4")
    if cap.initialize() == True:
        while True:
            ret, frame = cap.getFrame()
            frame = cv2.resize(frame, (640, 640))
            cv2.imshow("img", frame)
            cv2.waitKey(10)
        cap.release()
