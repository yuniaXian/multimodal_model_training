from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
import json
import random
from torchvision import datasets


model_name="my_model"
processor = ViTImageProcessor.from_pretrained(model_name)
#因为我们改变模型结构，所以下面的加载方式失效
#model = ViTForImageClassification.from_pretrained(model_name)
#直接用pytorch的方式加载
model=torch.load("my_model\\pytorch_model.bin").to("cuda:0") 
test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False,download=True)
data=test_dataset.data
targets=test_dataset.targets
right=0
count=0
for i,image in enumerate(data):
    inputs = processor(images=image, return_tensors="pt").to("cuda:0") 
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    if predicted_class_idx==targets[i]:
        right+=1
    count+=1
    print (count,len(targets))

print (right/count)

