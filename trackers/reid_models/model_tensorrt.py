import torch
import cv2
import numpy as np
import tensorrt as trt
from collections import OrderedDict, namedtuple


class Engine(object):
    def __init__(self, model_path, device=0):
        self.setup_device(device)
        self.fp16 = False
        self.setup_model(model_path)

    def setup_device(self, device):
        if not torch.cuda.is_available() or device == 'cpu':
            raise RuntimeError("CUDA device is required but not available or 'cpu' was explicitly selected.")
        elif isinstance(device, int):
            self.device = f'cuda:{device}'
        else:
            self.device = 'cuda:0'
    def eval(self):
        """ This method clone method eval() for pytorch model"""
        return

    def setup_model(self, model_path):
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()

        self.is_trt10 = not hasattr(self.model, "num_bindings")
        num = range(self.model.num_io_tensors) if self.is_trt10 else range(self.model.num_bindings)
        for i in num:
            if self.is_trt10:
                name = self.model.get_tensor_name(i)
                dtype = trt.nptype(self.model.get_tensor_dtype(name))
                is_input = self.model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input:
                    if -1 in tuple(self.model.get_tensor_shape(name)):
                        self.dynamic = True
                        self.context.set_input_shape(name, tuple(self.model.get_tensor_profile_shape(name, 0)[1]))
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_name = name
                shape = tuple(self.context.get_tensor_shape(name))
            else: # TensorRT < 10.0
                name = self.model.get_binding_name(i)
                dtype = trt.nptype(self.model.get_binding_dtype(i))
                is_input = self.model.binding_is_input(i)
                if self.model.binding_is_input(i):
                    if -1 in tuple(self.model.get_binding_shape(i)):
                        self.dynamic = True
                        self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0, i)[1]))
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_name = name
                shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def __call__(self, batch):
        dtype = torch.float16 if self.fp16 else torch.float32
        batch = batch.to(dtype)
        features = self._predict(batch) # raw feature, without L2-normalization
        return features

    def _predict(self, im_batch: torch.Tensor):
        im_batch = im_batch.to(self.device)
        if self.dynamic and im_batch.shape != self.bindings['input'].shape:
            if self.is_trt10:
                self.context.set_input_shape('input', im_batch.shape)
                self.bindings['input'] = self.bindings['input']._replace(shape=im_batch.shape)
                name = self.output_name
                self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
            else:
                i = self.model.get_binding_index('input')
                self.context.set_binding_shape(i, im_batch.shape)
                self.bindings['input'] = self.bindings['input']._replace(shape=im_batch.shape)
                name = self.output_name
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings['input'].shape
        assert im_batch.shape == s, f"input size {im_batch.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['input'] = int(im_batch.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings[self.output_name].data
        return y

