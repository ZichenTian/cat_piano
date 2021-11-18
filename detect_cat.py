import os
import sys
import copy
import cv2
import math
import numpy as np

from pyppl import nn as pplnn
from pyppl import common as pplcommon

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

g_pplnntype2numpytype = {
    pplcommon.DATATYPE_INT8 : np.int8,
    pplcommon.DATATYPE_INT16 : np.int16,
    pplcommon.DATATYPE_INT32 : np.int32,
    pplcommon.DATATYPE_INT64 : np.int64,
    pplcommon.DATATYPE_UINT8 : np.uint8,
    pplcommon.DATATYPE_UINT16 : np.uint16,
    pplcommon.DATATYPE_UINT32 : np.uint32,
    pplcommon.DATATYPE_UINT64 : np.uint64,
    pplcommon.DATATYPE_FLOAT16 : np.float16,
    pplcommon.DATATYPE_FLOAT32 : np.float32,
    pplcommon.DATATYPE_FLOAT64 : np.float64,
    pplcommon.DATATYPE_BOOL : bool,
}

def RegisterEngines():
    engines = []

    x86_options = pplnn.X86EngineOptions()
    x86_engine = pplnn.X86EngineFactory.Create(x86_options)

    engines.append(pplnn.Engine(x86_engine))
    return engines

class ModelRunner(object):
    def __init__(self, model_path):
        self.__initialize(model_path)
    
    def __initialize(self, model_path):
        engines = RegisterEngines()
        if len(engines) == 0:
            raise Exception('failed to register engines')

        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(model_path, engines)
        if not runtime_builder:
            raise Exception('failed to create runtime from file: %s' % (model_path))

        self.runtime = runtime_builder.CreateRuntime()
        if not self.runtime:
            raise Exception('failed to create runtime')
    
    def get_input_tensor_shape(self):
        return self.runtime.GetInputTensor(0).GetShape().GetDims()
    
    def forward(self, input):
        if not self.runtime:
            raise Exception('runtime not created')
        
        tensor = self.runtime.GetInputTensor(0)
        shape = tensor.GetShape()
        np_data_type = g_pplnntype2numpytype[shape.GetDataType()]
        dims = shape.GetDims()

        status = tensor.ConvertFromHost(input)
        if status != pplcommon.RC_SUCCESS:
            raise Exception('failed to set input data')
        
        status = self.runtime.Run()
        if status != pplcommon.RC_SUCCESS:
            raise Exception('failed to run')
        
        status = self.runtime.Sync()
        if status != pplcommon.RC_SUCCESS:
            raise Exception('failed to sync')
        
        out_datas = {}
        for i in range(self.runtime.GetOutputCount()):
            tensor = self.runtime.GetOutputTensor(i)
            tensor_name = tensor.GetName()
            tensor_data = tensor.ConvertToHost()
            if not tensor_data:
                raise Exception('failed to get output ' + tensor_name)
            
            out_data = np.array(tensor_data, copy=False)
            out_datas[tensor_name] = copy.deepcopy(out_data)
        
        return out_datas

class CatDetector(object):
    def __init__(self, model_path):
        self.model_runner = ModelRunner(model_path)
    
    def generate_proposals(self, output, anchors, stride, thresh):
        results = []

        num_grid_x = output.shape[2]
        num_grid_y = output.shape[3]
        num_classes = output.shape[4]
        for q, anchor in enumerate(anchors):
            anchor_w, anchor_h = anchor
            for i in range(num_grid_y):
                for j in range(num_grid_x):
                    feat = output[0, q, i, j]
                    class_score = feat[5]
                    box_score = feat[4]
                    confidence = sigmoid(box_score) * sigmoid(class_score)
                    if confidence >= thresh:
                        dx = sigmoid(feat[0])
                        dy = sigmoid(feat[1])
                        dw = sigmoid(feat[2])
                        dh = sigmoid(feat[3])

                        pb_cx = (dx * 2 - 0.5 + j) * stride
                        pb_cy = (dy * 2 - 0.5 + i) * stride

                        pb_w = math.pow(dw * 2, 2) * anchor_w
                        pb_h = math.pow(dh * 2, 2) * anchor_h

                        x0 = pb_cx - pb_w * 0.5
                        y0 = pb_cy - pb_h * 0.5
                        x1 = pb_cx + pb_w * 0.5
                        y1 = pb_cy + pb_h * 0.5

                        results.append([x0, y0, x1, y1, 0, confidence])

        return results
    
    def calc_iou(self, b0, b1):
        w0 = b0[2] - b0[0]
        h0 = b0[3] - b0[1]
        w1 = b1[2] - b1[0]
        h1 = b1[3] - b1[1]

        xc0 = (b0[0] + b0[2]) / 2
        yc0 = (b0[1] + b0[3]) / 2
        xc1 = (b1[0] + b1[2]) / 2
        yc1 = (b1[1] + b1[3]) / 2

        wi = max((w0 + w1) / 2 - abs(xc0 - xc1), 0)
        hi = max((h0 + h1) / 2 - abs(yc0 - yc1), 0)
        wu = w0 + w1 - wi
        hu = h0 + h1 - hi

        return wi * hi / (wu * hu)
    
    def nms(self, proposals, iou_thresh):
        sorted_proposals = sorted(proposals, key=lambda proposals : proposals[5], reverse=True)
        if len(proposals) == 0:
            return []
        filtered_proposals = [sorted_proposals[0]]
        for p in sorted_proposals[1:]:
            filtered = False
            for fp in filtered_proposals:
                if self.calc_iou(p, fp) >= iou_thresh:
                    filtered = True
                    break
            if not filtered:
                filtered_proposals.append(p)

        return filtered_proposals
    
    def run(self, img):
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = img.astype(dtype = np.float32)
        img = np.ascontiguousarray(img)
        img /= 255
        img = np.expand_dims(img, axis=0)
        
        outputs = self.model_runner.forward(img)
        output = outputs['output']

        thresh = 0.02

        proposals = []
        output = output[0]
        for xc, yc, w, h, box_score, cls_score in output:
            confidence = box_score * cls_score
            if confidence >= thresh:
                proposals.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + w / 2, 0, confidence])

        proposals = self.nms(proposals, thresh)
        if len(proposals) == 0:
            return []
        return proposals[:1]

if __name__ == '__main__':
    cat_detector = CatDetector('./catDetectorOp11.onnx')
    img = cv2.imread('./cat1.jpg')
    proposals = cat_detector.run(img)
    resized_img = cv2.resize(img, (640, 640))
    for x0, y0, x1, y1, _, _ in proposals:
        cv2.rectangle(resized_img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    cv2.imshow('img', resized_img)
    cv2.waitKey()

