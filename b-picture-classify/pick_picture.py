# -*- coding: utf-8 -*-
'''
Author: winddy
'''
import os
import shutil
import numpy as np


# 文件夹路径
file_path = './MPEG7-jpg/'

file_path_arr = []
for root, dirs, files in os.walk(file_path):
    files.sort()
    for file in files:
        # file_name = os.path.join(root, file)
        #file_path_arr.append(file_name)
        file_path_arr.append(file)

# 行是类别  列是每一个类别的数目
file_path_arr = np.reshape(file_path_arr, [-1, 20])
# 存储随机抽取的图片， 用于测试
output = []
for class_files in file_path_arr:
    choice = np.random.choice(class_files, size=1)
    output.append(choice)


# 把随机选取的文件移动到文件夹 test-jpg, 其余的移动到 train-jpg
train_file = './train-jpg/'
# 1： 先把MPEG7-jpg 的图片全部复制到文件夹 train-jpg中
for root, dirs, files in os.walk(file_path):
    for file in files:
        source_file = os.path.join(root, file)
        target_file = os.path.join(train_file, os.path.basename(source_file))
#         # 复制文件
        shutil.copy(source_file, target_file)

# 2： 再把文件夹train-jpg 中的且属于output中的图片移动到文件夹 test-jpg
test_file = './test-jpg/'
for root, dirts, files in os.walk(train_file):
    for file in files:
        # 如果文件要移动到文件夹 test-jpg
        if file in output:
            source_file = os.path.join(root, file)
            target_file = os.path.join(test_file, file)
            print(source_file+"--"+target_file)
            shutil.move(source_file, target_file)

# print(output)






