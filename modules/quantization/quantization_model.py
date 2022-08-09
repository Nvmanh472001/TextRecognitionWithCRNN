import torch
import torch.nn as nn

class QuantizationVGG(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizationVGG, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32
    
    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x
    

class QuantizationOps():
    def __init__(self, model, config, **kwargs):
        self.model = model
        self.config = config['quantization']
        
        if self.config['fuse_layers'] is None:
            fuse_layers = [['0', '1'], ['3', '4'], ['6', '7'], \
                        ['8', '9'], ['18', '19'], \
                        ['11', '12', '13'], ['14', '15', '16']]
        else:
            fuse_layers = self.config['fuse_layers']
        
        self.backend = self.config['backend'] if self.config['backend'] \
                                                else 'fbgemm'
        
        self.fuse_layers = fuse_layers
        fused_model = self.fused_layers()
        
        self.quantized_model = QuantizationVGG(model_fp32=fused_model)
        
        self.load_config()
        self.prepare_quantization()
        
    def fused_layers(self):
        '''
            The model has to be switched to training mode before any layer fusion.
            Otherwise the quantization aware training will not work correctly.
            Fuse the model in place rather manually.
            params: |
                - layers: list layers to fuse
        '''
        
        fused_model = self.model
        fused_model.train()
        
        # fuse layers:
        for m in fused_model.modules():
            if type(m) == nn.Sequential:
                for layer in self.fuse_layers:
                    torch.quantization.fuse_modules(m, layer, inplace=True)
        
        return fused_model
        
    def load_config(self):
        '''
            Setup quantization configurations
        '''
        quantization_config = torch.quantization.get_default_qconfig(self.backend)
        self.quantized_model.qconfig = quantization_config
        print(self.quantized_model.qconfig)

    def prepare_quantization(self):
        # Prepare quantized model before training
        torch.quantization.prepare_qat(self.quantized_model, inplace=True)
        
    def convert2model(self):
        # convert quantized model after training
        torch.quantization.convert(self.quantized_model, inplace=True)
