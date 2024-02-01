import json
import os
files = os.listdir("images/val2017/val2017")
print (files)
with open("text.txt",encoding="utf-8") as f:
    lines=[eval(s.strip()) for s in f.readlines()]
lines=[s for s in lines if s['label']==1 and s['img'] in set(files)]
print (len(lines))