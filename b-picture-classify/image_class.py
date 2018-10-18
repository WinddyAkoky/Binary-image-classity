# -*- coding: utf-8 -*-
'''
Author: winddy
'''

import os
import shutil


# 目标文件夹
files_path = './test-jpg/'

classes = []

for root, dirs, files in os.walk(files_path):
    count = 0
    first_class = True
    files.sort()
    # 遍历所有图片
    for file in files:
        # 每19张图片是一个类别
        source_path = os.path.join(root, file)
        if count < 19:
            # 分割出类别名称
            name = file.split('-')[0]
            # 如果类别是没出现过的，添加如类别集合
            if first_class:
                classes.append(name)
                first_class = False

            # 目标路径
            target_path = files_path + name + '/'
            # 如果目标路径不存在，则创建
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            # 移动图片到目标目录
            shutil.move(source_path, target_path+file)
        else:
            count = 0
            first_class = True











