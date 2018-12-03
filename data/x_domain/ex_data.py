#*-coding: UTF-8 -*-

import os
import sys
import shutil

count=0
for dirname in os.listdir(os.path.join('lfw_funneled')):
    for file in os.listdir(os.path.join('lfw_funneled',dirname)):
        print(file)
        count+=1
        if file.split('.')[-1]=='jpg':
            shutil.move(os.path.join('lfw_funneled',dirname,file),
                        os.path.join('lfw_funneled'))
    try:
        os.removedirs(os.path.join('lfw_funneled',dirname))
    except:pass

# print(count)