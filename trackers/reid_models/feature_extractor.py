import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import tensorrt as trt
from collections import OrderedDict, namedtuple
from .models import load_model


class Extractor(object):
    def __init__(self, model_path, pytorch_half=False,
                 device='cpu', size=(64, 64)):
        self.setup_device(device)
        self.size = size

        # exception
        if 'deepsort' in model_path:
            print('Original deepsort reid with default shape - (128, 64)')
            self.size = (128, 64)
        else:
            model_name = model_path.split('/')[-1]
            model_name = model_name.split('.')[0]
            print(f'ReID {model_name} with shape - {self.size}')
        self.pytorch_half = pytorch_half  # only for pytorch weights
        self.fp16 = False
        self.setup_model(model_path)
        print(f'half - {self.fp16}')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup_device(self, device):
        if not torch.cuda.is_available() or device == 'cpu':
            self.device = 'cpu'
        elif isinstance(device, int):
            self.device = f'cuda:{device}'
        else:
            self.device = 'cuda:0'

    def setup_model(self, model_path):
        postfix = model_path.split('.')[-1]
        if postfix in ['pt', 'pth']:    # pytorch
            self.framework = 'pytorch'
            self._setup_model_pytorch(model_path)
        elif postfix in ['onnx']:
            self.framework = 'onnx'
            self._setup_model_onnx(model_path)
        elif postfix in ['engine', 'trt']:
            self.framework = 'tensorrt'
            if self.device == 'cpu':    # must use GPU
                self.device = 'cuda:0'
            self._setup_model_tensorrt(model_path)
        else:
            print('invalid model')

    def _setup_model_pytorch(self, model_path):
        self.net = load_model(model_path=model_path, reid=True)
        self.fp16 = self.pytorch_half or self.fp16
        self.net.half() if self.fp16 else self.net.float()
        self.net.eval()
        self.net.to(self.device)

    def _setup_model_onnx(self, model_path):    # TODO: setup model onnx on GPU
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if 'cuda' in self.device and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.net = ort.InferenceSession(model_path, providers=providers)
        inp = self.net.get_inputs()[0] #list lenght 1
        if inp.type == "tensor(float16)":
            self.fp16 = True


    def _setup_model_tensorrt(self, model_path):
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


    def _preprocess(self, im_crops):
        def _resize(im, size):  # size := (HxW)
            return cv2.resize(im, (size[1], size[0]))   # cv2.resize expects (width, height)
        dtype = torch.float16 if self.fp16 else torch.float32
        im_batch = torch.cat([
            self.transform(_resize(im, self.size)).unsqueeze(0) 
            for im in im_crops
        ], dim=0).to(dtype)

        return im_batch

    def __call__(self, im_crops):

        im_batch = self._preprocess(im_crops)
        if self.framework == 'pytorch':
            features = self._predict_pytorch(im_batch)
        elif self.framework == 'onnx':
            features = self._predict_onnx(im_batch)
        elif self.framework == 'tensorrt':
            features = self._predict_tensorrt(im_batch)
        features = F.normalize(features.to(torch.float32), p=2, dim=1)

        return features.cpu().numpy()

    def _predict_pytorch(self, im_batch: torch.Tensor):
        with torch.no_grad():
            features = self.net(im_batch.to(self.device)) # raw features, without L2 normalization
        return features

    def _predict_onnx(self, im_batch: torch.Tensor):
        features = self.net.run(None, {'input': im_batch.cpu().numpy()})  # list
        return torch.from_numpy(np.concatenate(features, axis=0))

    def _predict_tensorrt(self, im_batch: torch.Tensor):
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

if __name__ == '__main__':
    import os
    img = cv2.imread(os.path.join(os.path.dirname(__file__), 'demo.jpg'))[:,:,(2,1,0)] # BGR to RGB
    extr = Extractor("checkpoint/visdrone-128-128/ckpt.pth")
    feature = extr([img])
    print(feature.shape)
