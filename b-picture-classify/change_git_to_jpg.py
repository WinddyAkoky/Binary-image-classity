# -*- coding: utf-8 -*-
'''
Author: winddy
'''
import os
from PIL import Image

# 元目录
file_path = './MPEG7/'
# 目标目录
target_path = './MPEG7-png/'


for root, dirs, files in os.walk(file_path):
    for file in files:
        img_path = root + file
        im = Image.open(img_path)

        try:
            while True:
                # 保存当前帧图片
                current = im.tell()
                im.save(target_path + file.split('.')[0] + '.png')
                im.seek(current+1)
        except EOFError:
            pass







