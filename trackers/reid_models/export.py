import torch
from models import *

class Exporter:
    """
    Export Pytorch model to other formats: ONNX or TensorRT
    """
    def __init__(self, half=True, int8=False, opt_size=(128, 128)):
        self.half = half
        self.int8 = int8
        self.dynamic = True  # always set dynamic=True when using model for task MOT with ReID
        self.opt_size = opt_size # optimal size of image input 
    def __call__(self, model: torch.nn.Module|str, output: str):

        output_type = output.split('.')[-1]
        if output_type not in ['trt', 'engine', 'onnx']:
            raise ValueError(f'Invalid format {output_type}. Valid export .trt and .engine')

        if output_type == 'onnx':
            self.export_onnx(model, output)

        if output_type in ['trt', 'engine']:
            input_type = model.split('.')[-1] if isinstance(model, str) else 'pth'       # 'pth' or 'onnx'
            if input_type == 'pth':
                print(f'Export to ONNX firstly')
                model_onnx = '.'.join(output.split('.')[:-1]) + '.onnx'
                self.export_onnx(model, model_onnx)
                self.export_engine(model_onnx, output)
                import os
                os.remove(model_onnx)
                print(f'removed {model_onnx} after creating {output}')
            else:
                self.export_engine(model, output)
    def export_onnx(self, model: torch.nn.Module|str, output: str):
        dynamic_axes={}
        dynamic_axes['input'] = {0: 'batch', 2: 'height', 3: 'width'}
        dynamic_axes['output'] = {0: 'batch'}
        if isinstance(model, str):
            model_path = model
            model = load_model(model_path, reid=True)
        model.to('cpu')
        model.eval()
        output_type = output.split('.')[-1]
        if output_type != 'onnx':
            raise ValueError(f"Invalid export format='{output_type}'. Valid formats are .onnx")

        dummy_input = torch.randn(1, 3, self.opt_size[0], self.opt_size[1])
        if self.half:
            dummy_input = dummy_input.half()
            model.half()
        torch.onnx.export(
            model,
            dummy_input,
            output,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=19
        )
        print(f'Exported to {output}')
    def export_engine(self, input_onnx: str, output: str, workspace=1):
        """
        workspace: size of temporary memory for TensorRT (GB)
        dynamic: dynamic size for batch dimension of input tensor
        """

        import tensorrt as trt
        is_trt10 = int(trt.__version__.split('.')[0]) >= 10
        logger = trt.Logger(trt.Logger.INFO)
        logger.min_severity = trt.Logger.Severity.VERBOSE # verbose

        # build engine
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace = int(workspace * (1<<30))
        if is_trt10 and workspace > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        elif workspace > 0: # TensorRT 7, 8
            config.max_workspace_size = workspace
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)

        # Read ONNX
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(input_onnx):
            raise RuntimeError(f'failed to load ONNX file: {input_onnx}')
        if self.dynamic:
            profile = builder.create_optimization_profile()
            h, w = self.opt_size
            min_shape = (1, 3, h, w)
            max_shape = (256, 3, h, w)
            opt_shape = (128, 3, h, w)
            profile.set_shape('input', min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)
        if self.int8:
            pass
        if self.half:
            config.set_flag(trt.BuilderFlag.FP16)

        # free CUDA memory
        torch.cuda.empty_cache()

        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(output, 'wb') as t:
            t.write(engine if is_trt10 else engine.serialize())
        print(f'Export to {output}')
if __name__ == '__main__':
    exporter = Exporter(half=False, image_shape=(128, 64))
    model_path = 'checkpoint/deepsort-reid-Adam-cosine/best_ckpt.pth'
    exporter(model_path, output='weights_onnx/deepsort_reid.onnx')
