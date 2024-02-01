from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
import json
import random
from torchvision import datasets
import numpy as np

model_name="vitbase"
#图像处理器，把像素0~255 变成0~1，并且归一化图像大小
processor = ViTImageProcessor.from_pretrained(model_name)
#ViT模型  to("cuda:0")把模型放在GPU里面
model = ViTForImageClassification.from_pretrained(model_name).to("cuda:0")
print (model)
path="E:\\code\\clip\\train_image\\000000000081.jpg"     
img = Image.open(path)
img=np.array(img)

#图片的归一化，0~255 -> -1,1
inputs = processor(images=img, return_tensors="pt").to("cuda:0") 

# #预测
outputs = model(**inputs)
#各类的概率
logits = outputs.logits
#去概率最大的index
predicted_class_idx = logits.argmax(-1).item()
#概率最大的index对应的label名称
label=model.config.id2label[predicted_class_idx]
print (label)