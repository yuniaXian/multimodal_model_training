from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from torch import nn
import numpy as np
import random
import torch
from torchvision import datasets
def get_data(batch_size,label_num,path):
    num=int(batch_size/len(label_num))
    data=[]
    for sub_data in label_num.values():
        if num<=len(sub_data):
            data.extend(random.sample(sub_data,num))
        else:
            data.extend(sub_data)

    images=[]
    labels=[]
    for p,label in data:
        try:  
            img_path="{}{}".format(path,p)
            img = Image.open(img_path)
            img2=np.array(img)
            img.close()
            if len(img2.shape)!=3:
                continue     
            images.append(img2)
            labels.append(label)
        except:
            continue
    return images,np.array(labels)
def no_contain(name,freeze_layer):
    for s in freeze_layer:
        if s in name:
            return False
    return True
def parser_data(path):
    with open(path,encoding="utf-8") as f:
        lines=[ eval(s.strip()) for s in f.readlines()]
    label_num={}
    for p,label in lines:
        if label not in label_num:
            label_num[label]=[]
        label_num[label].append([p,label])
    return label_num
def cal_right_rate(val_data):
    right=0
    images=val_data.data
    labels=val_data.targets
    data=random.sample([s for s in zip(images,labels)],500)
    images,labels=zip(*data)
    count=len(labels)
    for i,val_image in  enumerate(images):
        inputs = processor(images=val_image, return_tensors="pt").to("cuda:0") 
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        if predicted_class_idx==labels[i]:
            right+=1
    return right/count


def cal_paras(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
import torchvision.transforms as transforms

transform = transforms.Compose( [transforms.PILToTensor()])
#只有100类
#加载自己的数据
train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True,transform=transform, download=True)

#训练集
train_dataset.data=np.array(train_dataset.data)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
 
#测试集
test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False,transform=transform, download=True)
processor = ViTImageProcessor.from_pretrained('vitbase')
model = ViTForImageClassification.from_pretrained('vitbase')
#对原生vit模型的参数进行冻结
for  name,p in model.named_parameters():
    #根据name自行指定想训练的层
    #print (name)
    p.requires_grad=True#False
#分类层：改变模型结构，把类别调整成我想要的
model.classifier=nn.Linear(in_features =768, out_features = 100)
model.to("cuda:0") 
#打印下参数量，非必要的
cal_paras(model)
#修改模型配置，分类，一次只有一个类
model.config.problem_type == "single_label_classification"
#改对应的配置，现在的类别是100
model.num_labels=100
#训练的一些参数配置
batch_size=256
learning_rate=1e-3
epochs=100
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
step=0
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(trainloader):
        step+=1
        #前向传播，计算loss
        inputs = processor(images=images, return_tensors="pt").to("cuda:0") 
        labels = labels.to(torch.int64) 
        loss= model(**inputs,return_dict=False,labels=labels)[0]
        #梯度下降法，训练所有requires_grad=True的参数
        optimizer.zero_grad() # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
        print (step,loss)
        loss.backward() # loss反向传播
        optimizer.step() # 反向传播后参数更新 
        #每十步，看下效果
        if step%10==0:
            right_rate=cal_right_rate(test_dataset)
            print (step,"准确率",right_rate)
            #torch.save(model, "my_model\\pytorch_model.bin")
#torch.save(model, "my_model\\pytorch_model.bin") 
