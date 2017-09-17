import tensorflow as tf
import numpy as np
import time
import pickle
import os
import platform
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import matplotlib.pyplot as plt
from PIL import Image
# from scipy.signal import convolve2d
import scipy.signal as sig
from sklearn import preprocessing

max_steps = 3000
batch_size = 128
data_dir = './cifar-10-batches-py'


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def loss(logits, labels):
    #      """Add L2Loss to all the trainable variables.
    #      Add summary for "Loss" and "Loss/avg".
    #      Args:
    #        logits: Logits from inference().
    #        labels: Labels from distorted_inputs or inputs(). 1-D tensor
    #                of shape [batch_size]
    #      Returns:
    #        Loss tensor of type float.
    #      """
    #      # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


###

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """

    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)

        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def to_grayscale(im, weights=np.c_[0.2990, 0.5870, 0.1140]):
    """
    取原始图像的RGB值的加权平均来将图片转换为灰阶，权重矩阵为tile
    """
    # 默认的 weights = array([[ 0.2989,  0.587 ,  0.114 ]])
    tile = np.tile(weights, reps=(im.shape[0], im.shape[1], 1))
    # assert( tile.shape == im.shape )
    return np.sum(tile * im, axis=2)
    # np.sum意味着沿某一轴求和，axis=2为第三维（0为第一维）
    # 整个乘法意味着由图像每个像素点的RGB 得到 (R*0.2989+ G*0.5870+ B*0.1140)灰阶值，图像的二维尺寸不变，而减为单通道。


def normalized(im, scale):
    im1 = []
    max = np.max(im)
    min = np.min(im)
    for x in im:
        xx = []
        for y in x:
            # z = float(y - min) / (max - min) * scale
            # xx.append(z)
            yy = []
            for z in y:
                z = float(z - np.min(im)) / (np.max(im) - np.min(im)) * 255.0
                yy.append(z)
            xx.append(yy)
        im1.append(xx)
    im1 = np.array(im1)
    return im1


# cifar10.maybe_download_and_extract()

# images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
#                                                             batch_size=batch_size)
# images_test, labels_test = cifar10_input.inputs(eval_data=True,
#                                                 data_dir=data_dir,
#                                                 batch_size=batch_size)
# images_train, labels_train = cifar10.distorted_inputs()
# images_test, labels_test = cifar10.inputs(eval_data=True)

images_train, labels_train, images_test, labels_test = load_CIFAR10(data_dir)
image_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name='input')
label_holder = tf.placeholder(tf.int32, [batch_size], name='label')

# logits = inference(image_holder)
'''
'''
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
logits_out = tf.nn.softmax(logits, name='logits_out')

loss = loss(logits, label_holder)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 0.72

top_k_op = tf.nn.in_top_k(logits, label_holder, 1, name='output')

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()
###

images_train = images_train[:49920]
label_batch = labels_train[:49920]
images_test = images_test[:9984]
label_test = images_test[:9984]

images_train_gray = []
# for image in images_train[:128]:
#     images_train_gray.append(np.reshape(to_grayscale(image),[32,32,3]))
images_train_gray = images_train[:128]
# image = Image.open('../cifar-10-batches-py/saveimage/airplane/34628967.png').convert('L')
# image = np.asarray(image)
# for i in range(batch_size):
#     images_train_gray.append(np.reshape(image,[32,32,1]))

norm1_output = sess.run(norm1, feed_dict={image_holder: np.array(images_train_gray),
                                          label_holder: label_batch[:128]})
norm2_output = sess.run(norm2, feed_dict={image_holder: np.array(images_train_gray),
                                          label_holder: label_batch[:128]})
# ctn = norm1_output.shape[1]*norm1_output.shape[2]*norm1_output.shape[3]


fig, axes = plt.subplots(16, 8)
plt.axis('off')
# plt.axes.get_yaxis().set_visible(False)
# plt.axes.get_yaxis().set_visible(False)

ctn = 0;
for i in range(1):
    img64 = Image.new('L', (16 * 8, 16 * 8), color='white')
    norm1_ouput_1 = norm1_output[0]
    norm1_ouput_1 = np.array(norm1_ouput_1).transpose([2, 0, 1])
    # norm1_ouput_1 = np.reshape(norm1_ouput_1,[64,16,16])

    for i in range(len(norm1_ouput_1)):
        image = norm1_ouput_1[i]
        # image = np.reshape(image,[16,16,1])
        max = np.max(image)
        if max > 0:
            print(max)
            scale = 255.0 / max
            image = image * scale
        row = int(i / 8)
        col = i % 8
        img11 = Image.fromarray(np.array(image, dtype='uint8'))
        img64.paste(img11, box=(row * 16, col * 16))
        axes[row, col].imshow(img11, cmap='gray')
        path = './static/image/' + str(ctn) + '.png'
        img11.save(path)
        ctn += 1;
        # img11.show()
    # plt.show()
    img64.show(title=str(i))

for i in range(1):
    img64 = Image.new('L', (16 * 8, 16 * 8))
    norm1_ouput_1 = norm2_output[0]
    norm1_ouput_1 = np.array(norm1_ouput_1).transpose([2, 0, 1])
    for i in range(len(norm1_ouput_1)):
        image = norm1_ouput_1[i]
        max = np.max(image)
        if max > 0:
            print(max)
            scale = 255.0 / max
            image = image * scale
        row = int(i / 8) + 8
        col = i % 8
        img11 = Image.fromarray(np.array(image, dtype='uint8'))
        img64.paste(img11, box=(row * 16, col * 16))
        axes[row, col].imshow(img11, cmap='gray')
        path = './static/image/' + str(ctn) + '.png'
        img11.save(path)
        ctn += 1;
        # img11.show()
    plt.show()
    img64.show(title=str(i))