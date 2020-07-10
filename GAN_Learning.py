import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# region Demo1
# https://github.com/j32u4ukh/Tensorflow-101/blob/master/notebooks/13_Generative_Adversarial_Network.ipynb

# %%
def weight_variable(shape, name):
    # 截尾常態分配
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)


mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

# Parameters
batch_size = 256
g_dim = 128

x_d = tf.placeholder(tf.float32, shape=[None, 784])
x_g = tf.placeholder(tf.float32, shape=[None, 128])

weights = {
    "w_d1": weight_variable([784, 128], "w_d1"),
    "w_d2": weight_variable([128, 1], "w_d2"),
    "w_g1": weight_variable([128, 256], "w_g1"),
    "w_g2": weight_variable([256, 784], "w_g2")
}

biases = {
    "b_d1": bias_variable([128], "b_d1"),
    "b_d2": bias_variable([1], "b_d2"),
    "b_g1": bias_variable([256], "b_g1"),
    "b_g2": bias_variable([784], "b_g2"),
}

var_d = [weights["w_d1"], weights["w_d2"], biases["b_d1"], biases["b_d2"]]
var_g = [weights["w_g1"], weights["w_g2"], biases["b_g1"], biases["b_g2"]]


# Build generator and discriminator networks
# 一個 GAN 裡包含了兩個神經網路，一個是 generator，
# 另一個是 discriminator．首先照著定義個別建立網路．
#
# 生成式網路 generator(z) 會接受一個 128 維的輸入 z 並經過兩層網路以後產生一個 784 維的輸出，這就是所謂假的 MNIST 資料．
#
# 判斷式網路 discrimination(x) 接受一個 784 維的輸入 x 並經過兩層網路以後產生一個 1 維的輸出，
# 而輸出經過 sigmoid 之後是一個 0 ~ 1 的數，代表著判斷式網路此輸出 x 是真實資料的機率．
def generator(z):
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights["w_g1"]), biases["b_g1"]))
    h_g2 = tf.nn.sigmoid(tf.add(tf.matmul(h_g1, weights["w_g2"]), biases["b_g2"]))
    return h_g2


def discriminator(x):
    h_d1 = tf.nn.relu(tf.add(tf.matmul(x, weights["w_d1"]), biases["b_d1"]))
    h_d2 = tf.nn.sigmoid(tf.add(tf.matmul(h_d1, weights["w_d2"]), biases["b_d2"]))
    return h_d2


# Build cost functions
# 在定義訓練 cost 函數的時候，和之前的神經網路比較不一樣的地方是它沒有一個標準的 loss 計算．
# 畢竟這個模型要判斷的問題是真或假的問題．
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# g_sample:生成器的 output
g_sample = generator(x_g)

# d_real:判別器 判斷 真實數據 的 output
d_real = discriminator(x_d)

# d_fake:判別器 判斷 生成數據 的 output
d_fake = discriminator(g_sample)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
g_loss = -tf.reduce_mean(tf.log(d_fake))

# Training
# 接下來使用 AdamOptimizer 來做訓練，其中提到了要先對 discriminator 更新參數再對 geneartor 更新．
# 因此需要在其中指定更新參數 var_list．
# 只更新 discriminator
d_optimizer = tf.train.AdamOptimizer(0.0005).minimize(d_loss, var_list=var_d)
# 只更新 generator parameters
g_optimizer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=var_g)


def plot(samples):
    # fig = plt.figure(figsize=(4, 4))
    plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')

    plt.show()


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for step in range(20001):
    batch_x = mnist.train.next_batch(batch_size)[0]
    _, d_loss_train = sess.run([d_optimizer, d_loss], feed_dict={x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
    _, g_loss_train = sess.run([g_optimizer, g_loss], feed_dict={x_g: sample_Z(batch_size, g_dim)})

    if step <= 1000:
        if step % 100 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % g_loss_train)
        if step % 1000 == 0:
            g_sample_plot = g_sample.eval(feed_dict={x_g: sample_Z(16, g_dim)})
            plot(g_sample_plot)
    else:
        if step % 1000 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % g_loss_train)
        if step % 2000 == 0:
            g_sample_plot = g_sample.eval(feed_dict={x_g: sample_Z(16, g_dim)})
            plot(g_sample_plot)
sess.close()
# endregion


# region Demo2：DCGAN
# https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/14_DCGAN_with_MNIST.ipynb
# 跟 GAN 不同的是，DCGAN 把 convolution 引進網路結構中．
# 在 discriminator 中輸入的圖像會經過層層 convolution 之後變成一個預測是否為真實圖片的機率；
# 在 generator 中會把輸入的 z 向量經過層層 deconvolution 輸出生成式的圖片．可以看到它的結構如下圖．
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)


batch_size = 256
g_dim = 100


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')


# Build convolutional discriminator and generator
# 這裡我們會建立生成式網路 (generator) 以及判斷式網路 (discriminator)

# Discriminator
# 在 discriminator 中，首先會輸入一個 None x 784 維的 MNIST 資料，首先會先把它變成一個二維的圖片向量也就是
# None x 28(width) x 28(height) x 1(channels)．接下來就是連續的使用 filter 作 convolution，其中 strides = 2，
# 使得每做完一次 convolution 的輸出長和寬就會變成一半．以下是輸入圖片的維度變化順序：
# 原始輸入維度 None x 28 x 28 x 1
# 經過第一個 5 x 5 convolution filter 後輸出維度為 None x 14 x 14 x 32
# 經過第二個 5 x 5 convolution filter 後輸出維度為 None x 7 x 7 x 64
# 經過一個 fully connected 層變成維度 None x 1 的判斷機率輸出

# Generator
# 在 generator 中，首先會輸入一個 None x 128 維從 noise 中取樣出的向量，首先經過一個全連結會把它擴展成
# None x 4*4*64，再把它 reshape 成一個 None x 4(width) x 4(height) x 64(channels) 的輸入
# (可以想像成把輸入向量變成一個 4x4 的圖像)．接下來就是經過一連串的 deconvolution，最後輸出圖片．以下是輸入的維度變化順序：

# 原始輸入取樣向量維度 None x 100
# 經過第一個 fully connected 層以及 reshape 變成維度 None x 4 x 4 x 64
# 經過第一個 5 x 5 deconvolution filter 後輸出維度為 None x 7 x 7 x 32
# 經過第二個 5 x 5 deconvolution filter 後輸出維度為 None x 14 x 14 x 16
# 經過第三個 5 x 5 deconvolution filter 後輸出維度為 None x 28 x 28 x 1
x_d = tf.placeholder(tf.float32, shape=[None, 784])
x_g = tf.placeholder(tf.float32, shape=[None, g_dim])
weights = {
    "w_d1": weight_variable([5, 5, 1, 32], "w_d1"),
    "w_d2": weight_variable([5, 5, 32, 64], "w_d2"),
    "w_d3": weight_variable([7 * 7 * 64, 1], "w_d3"),

    "w_g1": weight_variable([g_dim, 4 * 4 * 64], "w_g1"),
    "w_g2": weight_variable([5, 5, 32, 64], "w_g2"),
    "w_g3": weight_variable([5, 5, 16, 32], "w_g3"),
    "w_g4": weight_variable([5, 5, 1, 16], "w_g4")
}

biases = {
    "b_d1": bias_variable([32], "b_d1"),
    "b_d2": bias_variable([64], "b_d2"),
    "b_d3": bias_variable([1], "b_d3"),
    "b_g1": bias_variable([4 * 4 * 64], "b_g1"),
    "b_g2": bias_variable([32], "b_g2"),
    "b_g3": bias_variable([16], "b_g3"),
    "b_g4": bias_variable([1], "b_g4"),
}

var_d = [weights["w_d1"],
         weights["w_d2"],
         weights["w_d3"],
         biases["b_d1"],
         biases["b_d2"],
         biases["b_d3"]]

var_g = [weights["w_g1"],
         weights["w_g2"],
         weights["w_g3"],
         weights["w_g4"],
         biases["b_g1"],
         biases["b_g2"],
         biases["b_g3"],
         biases["b_g4"]]


def generator(z):
    # 100 x 1
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights["w_g1"]), biases["b_g1"]))
    # -1 x 4*4*128
    h_g1_reshape = tf.reshape(h_g1, [-1, 4, 4, 64])

    output_shape_g2 = tf.stack([tf.shape(z)[0], 7, 7, 32])
    h_g2 = tf.nn.relu(tf.add(deconv2d(h_g1_reshape, weights["w_g2"], output_shape_g2), biases["b_g2"]))

    output_shape_g3 = tf.stack([tf.shape(z)[0], 14, 14, 16])
    h_g3 = tf.nn.relu(tf.add(deconv2d(h_g2, weights["w_g3"], output_shape_g3), biases["b_g3"]))

    output_shape_g4 = tf.stack([tf.shape(z)[0], 28, 28, 1])
    h_g4 = tf.nn.tanh(tf.add(deconv2d(h_g3, weights["w_g4"], output_shape_g4), biases["b_g4"]))

    return h_g4


def discriminator(x):
    x_reshape = tf.reshape(x, [-1, 28, 28, 1])
    # 28 x 28 x 1
    h_d1 = tf.nn.relu(tf.add(conv2d(x_reshape, weights["w_d1"]), biases["b_d1"]))
    # 14 x 14 x 32
    h_d2 = tf.nn.relu(tf.add(conv2d(h_d1, weights["w_d2"]), biases["b_d2"]))
    # 7 x 7 x 64
    h_d2_reshape = tf.reshape(h_d2, [-1, 7 * 7 * 64])
    h_d3 = tf.nn.sigmoid(tf.add(tf.matmul(h_d2_reshape, weights["w_d3"]), biases["b_d3"]))
    return h_d3


# Build cost function
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


g_sample = generator(x_g)
d_real = discriminator(x_d)
d_fake = discriminator(g_sample)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
g_loss = -tf.reduce_mean(tf.log(d_fake))

# 只更新 discriminator
d_optimizer = tf.train.AdamOptimizer(0.0001).minimize(d_loss, var_list=var_d)

# 只更新 generator parameters
g_optimizer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=var_g)


# Training
def plot(samples):
    # fig = plt.figure(figsize=(4, 4))
    plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')

    plt.show()


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for step in range(1, 20001):
    batch_x = mnist.train.next_batch(batch_size)[0]
    _, d_loss_train = sess.run([d_optimizer, d_loss], feed_dict={x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
    _, g_loss_train = sess.run([g_optimizer, g_loss], feed_dict={x_g: sample_Z(batch_size, g_dim)})

    if step <= 1000:
        if step % 100 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % g_loss_train)
        if step % 1000 == 0:
            g_sample_plot = g_sample.eval(feed_dict={x_g: sample_Z(16, g_dim)})
            plot(g_sample_plot)
    else:
        if step % 1000 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % g_loss_train)
        if step % 2000 == 0:
            g_sample_plot = g_sample.eval(feed_dict={x_g: sample_Z(16, g_dim)})
            plot(g_sample_plot)
# endregion


# region Demo3
# https://medium.com/@gau820827/%E6%95%99%E9%9B%BB%E8%85%A6%E7%95%AB%E7%95%AB-
# %E5%88%9D%E5%BF%83%E8%80%85%E7%9A%84%E7%94%9F%E6%88%90%E5%BC%8F%E5%B0%8D%E6%8A%97%E7%B6%B2%E8%B7%AF-
# gan-%E5%85%A5%E9%96%80%E7%AD%86%E8%A8%98-tensorflow-python3-dfad71662952
mnist = input_data.read_data_sets("data/MNIST_data/")
sample_image = mnist.train.next_batch(1)[0]
print(sample_image.shape)

sample_image = sample_image.reshape([28, 28])
plt.imshow(sample_image, cmap='Greys')


def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4


def generator(z, batch_size, z_dim):
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim / 2], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim / 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim / 2, z_dim / 4], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim / 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim / 4, 1], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4


z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

generated_image_output = generator(z_placeholder, 1, z_dimensions)
z_batch = np.random.normal(0, 1, [1, z_dimensions])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    generated_image = sess.run(generated_image_output,
                               feed_dict={z_placeholder: z_batch})
    generated_image = generated_image.reshape([28, 28])
    plt.imshow(generated_image, cmap='Greys')

tf.reset_default_graph()
batch_size = 50

# z_placeholder is for feeding input noise to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')

# x_placeholder is for feeding input images to the discriminator
x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x_placeholder')

# Gz holds the generated images
Gz = generator(z_placeholder, batch_size, z_dimensions)

# Dx will hold discriminator prediction probabilities for the real MNIST images
Dx = discriminator(x_placeholder)

# Dg will hold discriminator prediction probabilities for generated images
Dg = discriminator(Gz, reuse_variables=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    if i % 100 == 0:
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

# Train generator and discriminator together
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)

    if i % 100 == 0:
        # Every 100 iterations, show a generated image
        print("Iteration:", i, "at", datetime.datetime.now())
        z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
        generated_images = generator(z_placeholder, 1, z_dimensions)
        images = sess.run(generated_images, {z_placeholder: z_batch})
        plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
        plt.show()

        # Show discriminator's estimate
        im = images[0].reshape([1, 28, 28, 1])
        result = discriminator(x_placeholder)
        estimate = sess.run(result, {x_placeholder: im})
        print("Estimate:", estimate)
# endregion
