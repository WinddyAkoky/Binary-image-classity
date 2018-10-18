# -*- coding: utf-8 -*-
'''
Author: winddy
'''
import tensorflow as tf
import numpy as np
import time


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


batch_size = 32
classes_num = 70
image_width = image_height = 224
# 输入
image_holder = tf.placeholder(tf.float32, [None, image_width, image_height, 3])
# 标签
label_holder = tf.placeholder(tf.int32, [None, ])

# 定义网络结构

# 第一层卷积 (224 x 1-> 112 x 32)
conv1 = tf.layers.conv2d(
    inputs=image_holder,
    filters=32,
    kernel_size=[5, 5],
    padding='SAME',
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

# 第二层卷积 (112 x 32 -> 56 x 64)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding='SAME',
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 第三层卷积(56 x 64 -> 28 x 128)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding='SAME',
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 第四层卷积 (28 x 128 -> 14 x 128)
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding='SAME',
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 14 * 14 * 128])

# 全连接层
dense1 = tf.layers.dense(
    inputs=re1,
    units=1024,
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
)
dense2 = tf.layers.dense(
    inputs=dense1,
    units=512,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
)
logits = tf.layers.dense(
    inputs=dense2,
    units=classes_num,
    activation=None,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
)


loss = tf.losses.sparse_softmax_cross_entropy(labels=label_holder, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), label_holder)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 迭代次数
n_epoch = 1000

train_imgs, train_labels = read_and_decode("flower_train.tfrecords")
test_imgs, test_labels = read_and_decode("flower_train.tfrecords")

img_batch, label_batch = tf.train.shuffle_batch([train_imgs, train_labels],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)
t_img_batch, t_label_batch = tf.train.shuffle_batch([test_imgs, test_labels],
                                                    batch_size=70, capacity=2000,
                                                    min_after_dequeue=1000)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    threads = tf.train.start_queue_runners(sess=sess)
    # 训练
    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for i in range(50):
            print(i)
            img, label = sess.run([img_batch, label_batch])
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={
                image_holder: img, label_holder: label
            })
            n_batch += 1
            train_loss += err
            train_acc += ac
        end_time = time.time()
        print('Epoch %d, train loss: %f, train acc: %f, time: %f' % (epoch,
                                                                     train_loss, train_acc, (end_time - start_time)))
        # validation
        t_img, t_lab = sess.run([t_img_batch, t_label_batch])
        err, ac = sess.run([loss, acc], feed_dict={
            image_holder: t_img, label_holder: t_lab
        })
        print('validation: %f' % ac)

















