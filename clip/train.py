from PIL import Image
import requests
from transformers import HfArgumentParser, TrainingArguments, Trainer, set_seed
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset, Dataset
from transformers.configuration_utils import PretrainedConfig
import json
import numpy as np
import random
import numpy as np
#from torch.utils.data import DataLoader,Dataset
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
def get_data(num,path):
    with open(path,encoding="utf-8") as f:
        lines=[ eval(s.strip()) for s in f.readlines()]
    lines=[s for s in lines  if len(s[1])>0 and len(s[1])<20]
    #random.shuffle(lines)
    images=[]
    texts=[]
    for p,text in lines:
        try:  
            path="train_image\\{}".format(p)
            img = Image.open(path)
            img2=np.array(img)
            img.close()
            if len(img2.shape)!=3:
                continue     
            images.append(img2)        
            texts.append(text)
            if len(texts)>=num:
                 break
        except:
            continue
    return images,texts



#images=[ Image.open("train_image\\{}".format(p)) for p in path_list]
#加载官方提供的clip
model = CLIPModel.from_pretrained("Clip").to("cuda:0")
#print (model)
processor = CLIPProcessor.from_pretrained("Clip")
#返回loss
model.return_loss=True
#冻结住图像的encoder，只训练文本部分text encoder
for name, param in model.vision_model.named_parameters():
        param.requires_grad=False
batch_size=200
learning_rate=1e-4
max_step=500000
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(max_step):
    #从训练样本中抽取batchsize张图片样本对
    images,texts=get_data(batch_size,"train_data")
    inputs = processor(text=texts, images=images,return_tensors="pt", padding=True ).to("cuda:0")      
    loss= model(**inputs,return_loss=True,return_dict=False)[0]
    optimizer.zero_grad() # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
    print (t,loss)
    loss.backward() # loss反向传播
    optimizer.step() # 反向传播后参数更新
#保存模型
#     if t%10==0:
#         torch.save(model.state_dict(), "my_clip\\pytorch_model.bin")
# torch.save(model.state_dict(), "my_clip\\pytorch_model.bin") 










