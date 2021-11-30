import torch
import torchvision.models as models
from torchvision import transforms
from torch.onnx import register_custom_op_symbolic
import os
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
import base64
import hashlib
from PIL import Image
from tqdm import tqdm
import smart_open
from sklearn.metrics.pairwise import cosine_similarity

"""register_custom_op method registers the custom reduction operation on torch.onnx"""
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
            "mydomain::reduction", layer_one, layer_two, layer_three, layer_four
        )

    register_custom_op_symbolic("mynamespace::reduction", my_reuduction, 9)


"""
Model defination/code
input : [B, C, H, W] (here... 1, 3, 416, 416)
output : (1, 512) 
"""
class myModel(torch.nn.Module):
  def __init__(self):
    super(myModel,self).__init__()

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

"""store image paths in list"""
images= sorted(os.listdir('../Challenge_images'))
img_paths = []
for idx in range(len(images)):
  img_paths.append(os.path.join('../Challenge_images', images[idx]))

data_transforms = transforms.Compose([
    transforms.Resize(440),
    transforms.CenterCrop(416),
    transforms.ToTensor()
])

"""reads image from given path and return Tensor image"""
def image_loader(image_name :str) -> torch.Tensor:
    image = Image.open(image_name)
    image = data_transforms(image).float()
    image = image.unsqueeze(0)
    return image

"""
returns 1 if cosine similarity is less than 0.75 
that is images are dissimilar
"""
def return_result(cosinesimilarity):
  simlarity_score = round(float(cosinesimilarity),4)
  if cosinesimilarity > 0.75:
    return 0
  else:
    return 1

"""dictionary to store total count of dissimilar images"""
test_keys = []
for i in range(len(img_paths)):
  test_keys.append(img_paths[i][20:].partition(".")[0])
dissimilar_img_record = {}
for keys in test_keys:
    dissimilar_img_record[keys] = 0

"""created instance of model and kept in eval model"""
myModel = myModel()
myModel.eval()

"""
iterate over every single image in folder 
and count cosine similarity with every other image present in folder
"""
for i in tqdm(range(0,len(img_paths))):
  x = image_loader(img_paths[i])
  op1 = myModel(x)
  op1 = op1.detach().numpy()
  key = img_paths[i][20:].partition(".")[0]
  for j in range(0,len(img_paths)):
    y = image_loader(img_paths[j])
    op2 = myModel(y)
    op2 = op2.detach().numpy()
    similarity = cosine_similarity(op1,op2)
    result = return_result(similarity)
    dissimilar_img_record[key] = dissimilar_img_record[key] + result

"""creating dataframe of obtained result and store in current directory"""
df = pd.DataFrame({'image name' : dissimilar_img_record.keys(), 'dis-similar images' : dissimilar_img_record.values() })
df.to_csv(r'./result.csv', index=False)
print(df.to_string(index=False))