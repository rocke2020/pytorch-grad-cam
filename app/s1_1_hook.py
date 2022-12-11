import re, random
from pathlib import Path
import pickle
import json
import os, sys, shutil
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch.optim as optim
from torch.utils import data
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
sys.path.append(os.path.abspath('.'))
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image


model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
image_path = 'examples/both.png'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
print(f'input_tensor.shape {input_tensor.shape}')


class SaveOutput(object):
    """
    docstring
    """
    def __init__(self) -> None:
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.detach().numpy())

    def clear(self):
        self.outputs = []


save_output = SaveOutput()
# 需要注意的是hook函数在使用后应及时删除，以避免每次都运行增加运行负载。
hook_handles = []

for layer in model.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

out = model(input_tensor)
print(len(save_output.outputs))
for output in save_output.outputs:
    print(output.shape)
    break
print(len(hook_handles))
print(type(hook_handles[0]))