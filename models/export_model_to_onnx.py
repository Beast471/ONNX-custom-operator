import torchvision.models as models
from torch.onnx import register_custom_op_symbolic
import torch
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
import base64
import hashlib
from PIL import Image
from tqdm import tqdm
import numpy as np
import smart_open
from PIL import Image
import torch
from torchvision import transforms

def register_custom_op() -> None:
    torch.ops.load_library(
        "../inference/build/lib.linux-x86_64-3.7/reduction_op.cpython-37m-x86_64-linux-gnu.so"
    )

    def my_reuduction(
        g: object,
        layer_one: torch.Tensor,
        layer_two: torch.Tensor,
        layer_three: torch.Tensor,
        layer_four: torch.Tensor,
    ) -> object:
        return g.op(
            "mynamespace::reduction", layer_one, layer_two, layer_three, layer_four
        )

    register_custom_op_symbolic("mynamespace::reduction", my_reuduction, 9)


def export_model():

  class CustomModel(torch.nn.Module):
    def __init__(self):
      super(CustomModel, self).__init__()
      
      register_custom_op()
      self.resnet18 = models.resnet18(pretrained=True) 
    
      #initail
      self.new_model1 = torch.nn.Sequential(*list(self.resnet18.children())[:4])
      #layer 1
      self.new_model2 = torch.nn.Sequential(*list(self.resnet18.children())[4:5])
      self.pool2 = torch.nn.AdaptiveAvgPool2d((1,1))
      #layer 2
      self.new_model3 = torch.nn.Sequential(*list(self.resnet18.children())[5:6])
      self.pool3 = torch.nn.AdaptiveAvgPool2d((1,1))
      #layer 3
      self.new_model4 = torch.nn.Sequential(*list(self.resnet18.children())[6:7])
      self.pool4 = torch.nn.AdaptiveAvgPool2d((1,1))
      #layer 4
      self.new_model5 = torch.nn.Sequential(*list(self.resnet18.children())[7:8])
      self.pool5 = torch.nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self,x):
      x = self.new_model1(x) 

      x = self.new_model2(x)
      layer2 = x
      layer2 = self.pool2(x) 
      layer2 = torch.reshape(layer2, (1, 64)) 

      x = self.new_model3(x) 
      layer3 = x
      layer3 = self.pool3(x)
      layer3 = torch.reshape(layer3, (1, 128)) 

      x = self.new_model4(x) 
      layer4 = x
      layer4 = self.pool4(x) 
      layer4 = torch.reshape(layer4, (1, 256)) 
      
      x = self.new_model5(x) 
      layer5 = x
      layer5 = self.pool5(x) 
      layer5 = torch.reshape(layer5, (1, 512)) 

      embedding = torch.ops.mynamespace.reduction(
                      layer2,
                      layer3,
                      layer4,
                      layer5,
                      )
      embedding = torch.reshape(embedding, (1, 512)) 

      return embedding 
  
  input_img = torch.randn((1, 3, 416, 416))
    
  inputs = (input_img)

  f = './ReductionResnetonnx.onnx'
  torch.onnx.export(CustomModel(), inputs, f,
                       opset_version=9,
                       example_outputs=None,
                       input_names=["input_img"], output_names=["Y"],
                       custom_opsets={"mynamespace": 9})

export_model()
 