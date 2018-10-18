# -*- coding: utf-8 -*-
'''
Author: winddy
'''
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 读取图片信息，存储到 tfrecord格式文件
def read_image_to_tfrecode(path, save_name):
    # source_file_path = './train-jpg/'
    source_file_path = path
    classes = []

    for dirname in os.listdir(source_file_path):
        classes.append(dirname)

    print(classes)
    # "flower_train.tfrecords"
    writer = tf.python_io.TFRecordWriter(save_name)

    # 为了保证标签与对应的数字不会混乱，我们对类别集合按自然排序的方法进行排序，这样在制作训练数据和
    # 测试数据时实现彼此中类别与数字对应关系是一样的
    classes.sort()
    for index, name in enumerate(classes):
        class_path = source_file_path + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img_raw = img.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())

    writer.close()


# read_image_to_tfrecode('./train-jpg/')
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)

    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    # label = tf.one_hot(label, 70, 1, 0)
    return image, label


# read_image_to_tfrecode('./train-jpg/', 'flower_train.tfrecords')
# imgs, labels = read_and_decode("flower_train.tfrecords")

img_batch, label_batch = tf.train.shuffle_batch([imgs, labels],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
# init = tf.initialize_all_variables()
#
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(5):
        img, label = sess.run([img_batch, label_batch])
        # print(img)
        print(label)
# print('dfdfd')

# read_image_to_tfrecode('./train-jpg/', 'flower_train.tfrecords')
# read_image_to_tfrecode('./test-jpg/', 'flower_test.tfrecords')
# imgs, labels = read_and_decode("flower_train.tfrecords")


























