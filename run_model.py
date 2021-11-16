import os
import sys
import copy
import numpy as np

from pyppl import nn as pplnn
from pyppl import common as pplcommon

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
        self.model_path = model_path
    
    def initialize(self):
        engines = RegisterEngines()
        if len(engines) == 0:
            raise Exception('failed to register engines')

        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(self.model_path, engines)
        if not runtime_builder:
            raise Exception('failed to create runtime')

        self.runtime = runtime_builder.CreateRuntime()
        if not self.runtime:
            raise Exception('failed to create runtime')
    
    def forward(self, input):
        if not self.runtime:
            raise Exception('runtime not created')
        
        tensor = self.runtime.GetInputTensor(0)
        shape = tensor.GetShape()
        np_data_type = g_pplnntype2numpytype[shape.GetDataType()]
        dims = shape.GetDims()

        in_data = np.zeros(dims, dtype=np_data_type)
        status = tensor.ConvertFromHost(in_data)
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

if __name__ == '__main__':
    model_runner = ModelRunner('./catDetectorOp11.onnx')
    model_runner.initialize()
    outputs = model_runner.forward(None)
    print(outputs['output'].shape)
        
