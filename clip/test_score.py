import copy
import random
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
def get_data(num,path):
    with open(path,encoding="utf-8") as f:
        lines=[ eval(s.strip()) for s in f.readlines()]
    lines=[s for s in lines  if len(s[1])>0 and len(s[1])<20]
    #random.shuffle(lines)
    images=[]
    texts=[]
    for p,text in lines:
        try:  
            path="val_image\\{}".format(p)
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
def cal_score(model,images,texts1,texts2,processor):
    result=[]
    for image,text1,text2 in zip(images,texts1,texts2):
        inputs = processor(text=[text1,text2], images=image, return_tensors="pt", padding=True).to("cuda:0")
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image 
        probs = logits_per_image.softmax(dim=1)
        score1,score2=probs.tolist()[0]
        #正样本的分数-负样本的分数
        score=score1-score2
        result.append(score)
    return sum(result)/len(result)
#从测试集里面读取一些数据
#images 和texts1是一一对应，为正样本
images,texts1=get_data(1000,"val_data")
#texts2是负样本
texts2=copy.deepcopy(texts1)
random.shuffle(texts2)
#训练好的模型
# 一个 图像-文本对 做正样本
#图像文本随机匹配 做负样本
model_name="my_clip"
model = CLIPModel.from_pretrained(model_name).to("cuda:0")
processor = CLIPProcessor.from_pretrained(model_name)
score=cal_score(model,images,texts1,texts2,processor)
#测试正样本的得分比负样本高多少
print ("model_trained",score)

#原生clip
model_name="Clip"
model = CLIPModel.from_pretrained(model_name).to("cuda:0")
processor = CLIPProcessor.from_pretrained(model_name)
score=cal_score(model,images,texts1,texts2,processor)
print ("model",score)