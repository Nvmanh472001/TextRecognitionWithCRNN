import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from model import Model
from utils import CTCLabelConverter
from dataset import AlignCollate
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class RecognizerPrediction(object):
    def __init__(self, opt, quantize=True) -> None:
        self.opt = opt
        
        model = Model(opt=opt)
        weights = torch.load(opt.saved_model, map_location=device)
        
        new_state_dict = OrderedDict()
        for key, value in weights.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        
        if quantize:
            try:
                torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
            except:
                pass
        
        self.model = model
        self.converter = model.Prediction
        
    
    