 # -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:28:31 2017

Neural Style Transfer with Keras 2.0.5
Based on: https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
Contains few improvements suggested in the paper Improving the Neural Algorithm of Artistic Style (http://arxiv.org/abs/1605.04603).

@author: SuperSun
"""

# to do: 下面我们会逐渐地把掩膜等其它特性加上，最后达到消化理解这篇杰作的目的。然后我们会写一篇论文，加入自己的理解。
import os

from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils.data_utils import get_file
import tensorflow as tf
# -----------------------------------------------------------------------------------------------------------------------

# 参数：内容图像名，样式图像名列表，样式图像权重，输出图像宽度，内容图像权重，TV权重，迭代次数，输出图像路径，运算设备
base_image_path = './Yc.jpg'
style_image_paths = ['./Ys.jpg']
style_weights = [1.0]
img_size = 200
content_weight = 0.001
total_variation_weight = 8.5e-5
num_iter = 30
output_path = './results'
device_type = '/cpu:0'

if not os.path.exists(base_image_path):
	raise RuntimeError('No content image detected in file path.')
for p in style_image_paths:
	if not os.path.exists(p):
		raise RuntimeError('No content image detected in file path.')
if not os.path.exists(output_path):
	os.mkdir(output_path)

# -----------------------------------------------------------------------------------------------------------------------

# 工具函数。将图像转换为VGG网络所使用的张量。
def preprocess_image(image_path, load_dims=False):
    global img_width, img_height, aspect_ratio
    img = imread(image_path, mode='RGB')
    if load_dims:       # 在读取图像后，按原始图像的长宽比缩放图像。否则就将图像直接缩放到img_width*img_height大小（这两个参数将在第1次运行时确定）
        aspect_ratio = float(img.shape[1]) / img.shape[0]
        img_width = img_size
        img_height = int(img_width * aspect_ratio)
    img = imresize(img, (img_width, img_height)).astype('float32')
    # RGB排列转换为BGR排列
    img = img[:, :, ::-1]   # L[::5]，代表（以从前到后的顺序）每5个值取1个值。那么L[::-1]就代表（以从后往前的顺序）每1个值取1个值，也即倒序
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    # 把3维数据img变成4维数据（添加最高维）
    img = np.expand_dims(img, axis=0)
    return img

# 工具函数。将VGG网络所使用的张量转换为图像。preprocess_image的逆函数。
def deprocess_image(x):
    x = x.reshape((img_width, img_height, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]       # RGB排列转换为BGR排列
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 工具函数：计算gram矩阵
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))  # K.permute_dimensions：重新排列x的3个维度。利如3维张量x的shape是(200, 284, 64)，现在就变成了(64, 200, 284)
    gram = K.dot(features - 1, K.transpose(features - 1))
    return gram

# 工具函数：计算合成图像与样式图像之间的style损失
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    return K.sum(K.square(S - C)) / (4. * (3 ** 2) * ((img_width * img_height) ** 2))

# 工具函数：计算合成图像的TV损失，用于保持合成图像的局部结构
def total_variation_loss(x):
    a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])   # 在这里，x是一个4维的张量。其中后3维是(高, 宽, 通道数)
    b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
# -----------------------------------------------------------------------------------------------------------------------

# 主处理脚本开始
with tf.device(device_type):   # 大图像将导致显存不够，需要改用CPU计算（'/cpu:0'）
# 1. 准备图像数据。这里的所有图像都是一个4维张量
    base_image = preprocess_image(base_image_path, True)
    style_reference_images = []
    for style_path in style_image_paths:
        style_reference_images.append(preprocess_image(style_path))
    combination_image = K.placeholder((1, img_width, img_height, 3))    # 合成图像作为一个张量，这个值我们并不知道，因此用占位符代替
    nb_style_images = len(style_reference_images)   # style图像数量
    nb_tensors = nb_style_images + 2                # 总图像数量
# 把3种图像放到一个list中。例如，这个list的元素个数为3，每个元素都是一个表示图像的4维张量
    image_tensors = [base_image]
    for style_image_tensor in style_reference_images:
        image_tensors.append(style_image_tensor)
    image_tensors.append(combination_image)
# 捏合上面的list变成一个4维张量的形式，作为keras的输入。
    input_tensor = K.concatenate(image_tensors, axis=0)     # input_tensor(0, :, :, :)代表内容图像，input_tensor(-1, :, :, :)代表合成图像，input_tensor(2:-1, :, :, :)代表style图像

# 2. 构建网络
    # (comm 1:)VGG网络不是接收一个图像吗？在这里，注意batch_shape参数的第1个分量nb_tensors。这个值正好可以使得：VGG网络对输入的所有图像“并行”执行相同的操作。
    ip = Input(tensor=input_tensor, batch_shape=(nb_tensors, img_width, img_height, 3))
# 2.1 手动建立每一层
    x = Convolution2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(ip)
    x = Convolution2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Convolution2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
    x = Convolution2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Convolution2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
    x = Convolution2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
    x = Convolution2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(x)
    x = Convolution2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
# 2.2 建立模型并读入预训练系数
    model = Model(ip, x)
    weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', cache_subdir='models')
    model.load_weights(weights)
# 2.3 获得每个“关键”层的信息。这些信息以字典方式组织。
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    shape_dict = dict([(layer.name, layer.output_shape) for layer in model.layers])

# 3. 计算输入x的loss和grads（符号表达）
    feature_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
# 3.1 计算content损失
    loss = 0.0
    layer_features = outputs_dict['conv2_2']    # 这里的layer_features张量的第1维为nb_tensors，正好与上面(comm 1:)注释部分的内容相吻合。
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[nb_tensors-1, :, :, :]
    loss += content_weight * K.sum(K.square(base_image_features - combination_features))
# 3.2 计算style损失（Improvement : Chained Inference without blurring）
    nb_layers = len(feature_layers) - 1     # 从conv1_1一直使用到conv5_2
    for i in range(nb_layers):
# 在第i层，style特征的计算方式如下：
        #####################################################################
        # 3.2.a：取出第i层合成图像的特征（第i层的网络输出）combination_features；取出第i层style图像集的特征style_reference_features
        #        分别计算合成图像与每个style图像的“style loss”，记为sl1
        # 3.2.b：取出第i+1层合成图像的特征combination_features；取出第i+1层style图像集的特征style_reference_features
        #        分别计算合成图像与每个style图像的“style loss”，记为sl2
        # 3.2.c：由sl1和sl2来计算本层合成图像与style图像集的style loss
        #####################################################################
        # a步
        layer_features = outputs_dict[feature_layers[i]]    # 取出第i个卷积层的输出特征（[内容图像；样式图像集；合成图像]三部分组成）
        shape = shape_dict[feature_layers[i]]
        combination_features = layer_features[nb_tensors-1, :, :, :]    # 分离出合成图像的特征
        style_reference_features = layer_features[1:nb_tensors-1, :, :, :]  # 分离出样式图像集的特征
        sl1 = []
        for j in range(nb_style_images):    # 对于每个样式图像，都计算其与合成图像之间的style特征损失
            sl1.append(style_loss(style_reference_features[j], combination_features))
        # b步
        layer_features = outputs_dict[feature_layers[i + 1]]
        shape = shape_dict[feature_layers[i + 1]]
        combination_features = layer_features[nb_tensors-1, :, :, :]
        style_reference_features = layer_features[1:nb_tensors-1, :, :, :]
        sl2 = []
        for j in range(nb_style_images):
            sl2.append(style_loss(style_reference_features[j], combination_features))
        for j in range(nb_style_images):
            # c步
            loss += (style_weights[j] / (2 ** (nb_layers - (i + 1)))) * (sl1[j] - sl2[j])
# 3.3 计算总的损失和梯度
    loss += total_variation_weight * total_variation_loss(combination_image)
    grads = K.gradients(loss, combination_image)

# 4. 合成图像优化求解。
# 4.1 现在已经有了loss和grads的符号表达了。但是下面的fmin_l_bfgs_b却要求关于输入合成图像的“函数形式”的loss和grads，因此需要再作换换
    f_output_loss = K.function([combination_image], [loss])
    f_output_grads = K.function([combination_image], grads)
    # 根据上面的辅助函数，生成fmin_l_bfgs_b所需要的函数
    def f_loss(x):
        x = x.reshape((1, img_width, img_height, 3))
        outs = f_output_loss([x])
        return outs[0]
    def f_grads(x):
        x = x.reshape((1, img_width, img_height, 3))
        outs = f_output_grads([x])
        return np.array(outs).flatten().astype('float64')
# 4.2 利用fmin_l_bfgs_b函数，通过将loss(x)最小化的方式，求得最优的x。maxfun的值会直接影响收敛速度（迭代的图像质量）
#     可以再一次地看到，这个算法和VGG网络真的没有太大的联系，它只是用来提供特征而已。
#     算法的本质是，根据变量x，可以得到一个loss函数f(x)（这个函数由合成图像x、内容图像以及样式图像的VGG输出特征决定），然后我们去优化求解f(x)的最小值。
    x = preprocess_image(base_image_path, True)
    for i in range(num_iter):
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(f_loss, x.flatten(), fprime=f_grads, maxfun=20)
        img = deprocess_image(x.copy())
        # 图像尺寸调整及中间结果保存
        img_ht = int(img_width * aspect_ratio)
        img = imresize(img, (img_width, img_ht), interp='bicubic')
        fname = output_path + "_at_iteration_%d.png" % (i + 1)
        imsave(fname, img)
        print("Current loss value: %d, Iteration %d completed in %ds." % (min_val/img_width/img_height, i+1, time.time() - start_time))
        


