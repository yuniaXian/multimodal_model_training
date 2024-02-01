from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
import torch


model_name="Clip"
model = CLIPModel.from_pretrained(model_name).to("cuda:0")

processor = CLIPProcessor.from_pretrained(model_name)
#从clip模型中获取文本模块
text_model=model.text_model
#从clip模型中，获取图像模块
vision_model=model.vision_model
path = "E:\\code\\clip\\val_image\\000000179765.jpg"
image = Image.open(path)

text_inputs = processor(text="一辆黑色本田摩托车 背着黑色的布贡迪座椅", return_tensors="pt", padding=True).to("cuda:0")
img_inputs=processor(images=image, return_tensors="pt").to("cuda:0")
#model.visual_projection
#获取文本向量，last_hidden_state 输出最后一层对应的向量
#第一个：对应的所有batchsize 第二个：对应的是整个向量
text_output=text_model(**text_inputs).last_hidden_state[:, 0, :]
#Vit的输出也是个向量序列
image_output=vision_model(**img_inputs).last_hidden_state[:, 0, :]
text_output=model.text_projection(text_output)
image_output=model.visual_projection(image_output)

print (text_output)
print (text_output.shape)
print (image_output)
print (image_output.shape)
 