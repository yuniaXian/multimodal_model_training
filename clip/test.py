from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
import torch


#model_name="Clip"
model_name="my_clip"
model = CLIPModel.from_pretrained(model_name).to("cuda:0")
print (model)
 
#model=torch.load("myclip2.pth").to("cuda:0")
processor = CLIPProcessor.from_pretrained(model_name)

path = "E:\\code\\clip\\val_image\\000000179765.jpg"
image = Image.open(path)

inputs = processor(text=["一辆黑色本田摩托车 背着黑色的布贡迪座椅","一张有喷泉的厕所的特写照片"], images=image, return_tensors="pt", padding=True).to("cuda:0")

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print (logits_per_image)
print (probs)