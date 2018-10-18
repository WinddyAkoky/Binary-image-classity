**这篇文章的主要目的是：**
1. 根据本地文件，生成自己的数据集。
2. 编写分类网络，实现二值图分类

**MEPG7数据集下载地址：http://www.dabi.temple.edu/~shape/MPEG7/dataset.html**
**全部代码：https://github.com/WinddyAkoky/Binary-image-classity**


# 一：建立数据集

## TFRecord 输入数据格式介绍
1. Tensorflow 提供了一种统一的格式来存储数据，这个格式就是 TFRecord， 方便数据的处理和计算。因为复杂的图像处理函数有可能降低训练速度，为了加速数据处理过程，TFRecord 提供了多线程机制加速处理我们的数据，且利用 TFRecord 能更有效的管理数据属性。
2. TFRecord 文件中的数据都是通过 tf.train.Example  Protocol Buffer 的格式存储的。以下代码给出了 tf.train.Example 的定义：
```
message Example{
  Features features = 1;
}

message Features {
  map<string, Feature> feature = 1;
}

message Feature {
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
}
```
**从以上代码可以看出 tf.train.Example 的数据结构是比较简洁的。 tf.train.Example 中包含了一个从属性名称到取值的字典。其中属性名称为一个字符串，属性的取值可以为字符串（BytesList）、实数列表（FloatList）或者整数列表（Int64List）。比如将一张解码前的图像存储为一个字符串，图像所对应的类别编号存储为整数列表**

## 数据集代码
根据上面的理解，我们现在一点点建立我们的数据集。
1. 首先，在本地要有一些数据集，如下图所示：
![image.png](https://upload-images.jianshu.io/upload_images/13326502-2575992b4e920ed2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
2. **上图是数据库 MPEG7的数据集，从数据库中下载到的是 git格式，现在把它转成png格式：**
```
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
```
3. **在把png图片转为jpg图片**
```
filer_path = './MPEG7-png'

for root, dirs, files in os.walk(filer_path):
    for file in files:
        absorlute_path = root + '/' + file

        file_split = file.split('.')

        img = Image.open(absorlute_path)

        img = img.convert('RGB')

        save_path = './MPEG7-jpg/'+file_split[0] + '.jpg'
        print(save_path)
        img.save(save_path)
```
**最后应该如下图所示：**
![image.png](https://upload-images.jianshu.io/upload_images/13326502-3a3a87d33f9c3aca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

4.  **下面把文件夹中所属统一类的图片移动到统一文件夹中，在这个过程中，每一个类别随机抽取一张图片作为测试集**
```
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
```
成功后应该如下：
![](https://upload-images.jianshu.io/upload_images/13326502-facd54591bdeb7aa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
**当然，还有一个文件夹 test-jpg，这个有就咩有贴出来**

![image.png](https://upload-images.jianshu.io/upload_images/13326502-742135dec68b3ca6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
文件夹名就是相应的类别，我们把这个作为标签
![image.png](https://upload-images.jianshu.io/upload_images/13326502-70e556a746bd5210.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

5. 已知我们的文件夹如上，对与训练集中的文件夹下，有70个子文件夹，分别表示一类，子文件夹下有19张图片。 现在我们要读取这些图片，并把这些图片存储为 TFRecord格式：
```
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

read_image_to_tfrecode('./train-jpg/', 'flower_train.tfrecords')
read_image_to_tfrecode('./test-jpg/', 'flower_test.tfrecords')
```
最后在文件夹下生成我们的数据：
![image.png](https://upload-images.jianshu.io/upload_images/13326502-ca8dc535b7d7fbb5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 二：建立网络并训练网络：
```
atch_size = 32
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
```

