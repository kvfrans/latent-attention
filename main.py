import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
from glob import glob

class LatentAttention():
    def __init__(self):

        self.n_hidden = 500
        self.n_z = 200
        self.batchsize = 100
        self.num_colors = 3
        self.img_dim = 64
        self.sequence_length = 10

        self.images = tf.placeholder(tf.float32, [None, self.img_dim, self.img_dim, self.num_colors])
        z_mean, z_stddev = self.encoder(self.images)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.recurrent_generation(guessed_z)

        self.images_flat = tf.reshape(self.images, [-1, self.img_dim*self.img_dim*self.num_colors])
        self.generated_images_flat = tf.reshape(self.generated_images, [-1, self.img_dim*self.img_dim*self.num_colors])

        self.generation_loss = tf.nn.l2_loss(self.images_flat - self.generated_images_flat)
        self.generation_loss = self.generation_loss / (self.img_dim*self.img_dim*self.num_colors)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.latent_loss = self.latent_loss / self.n_z
        # self.latent_loss = self.latent_loss * 0.00

        self.cost = tf.reduce_mean(self.generation_loss * 10 + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def encoder(self, input_images):
        with tf.variable_scope("encoder"):

            e_bn2 = batch_norm(name="e_bn2")
            e_bn3 = batch_norm(name="e_bn3")

            h1 = lrelu(conv2d(input_images, self.num_colors, 64, "e_h1")) # 64x64x3 -> 32x32x16
            h2 = lrelu(e_bn2(conv2d(h1, 64, 128, "e_h2"))) # 32x32x16 -> 16x16x32
            h3 = lrelu(e_bn3(conv2d(h2, 128, 256, "e_h3"))) # 16x16x32 -> 8x8x64
            flattened = tf.reshape(h3,[self.batchsize, 8*8*256])
            flattened = tf.nn.dropout(flattened, 0.5)

            w_mean = dense(flattened, 8*8*256, self.n_z, "w_mean")
            w_stddev = dense(flattened, 8*8*256, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):

            g_bn1 = batch_norm(name="g_bn1")
            g_bn2 = batch_norm(name="g_bn2")
            g_bn3 = batch_norm(name="g_bn3")

            z_develop = dense(z, self.n_z, 8*8*256, scope='z_matrix')
            z_matrix = tf.nn.relu(g_bn1(tf.reshape(z_develop, [self.batchsize, 8, 8, 256])))
            h1 = tf.nn.relu(g_bn2(conv_transpose(z_matrix, [self.batchsize, 16, 16, 128], "g_h1")))
            h2 = tf.nn.relu(g_bn3(conv_transpose(h1, [self.batchsize, 32, 32, 64], "g_h2")))
            h3 = conv_transpose(h2, [self.batchsize, 64, 64, 3], "g_h3")
            final = tf.nn.sigmoid(h3)

        return final

    # decoder over multiple timesteps
    def recurrent_generation(self, z):
        with tf.variable_scope("generation"):

            self.cs = [0] * self.sequence_length

            for t in xrange(self.sequence_length):

                c_prev = tf.zeros((self.batchsize, self.img_dim, self.img_dim, self.num_colors)) if t == 0 else self.cs[t-1]

                z_develop = dense(z, self.n_z, 8*8*256, scope='z_matrix')
                z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 8, 8, 256]))
                h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 16, 16, 128], "g_h1"))
                h2 = tf.nn.relu(conv_transpose(h1, [self.batchsize, 32, 32, 64], "g_h2"))
                h3 = conv_transpose(h2, [self.batchsize, 64, 64, 3], "g_h3")
                self.cs[t] = c_prev + h3

        return tf.nn.sigmoid(self.cs[-1])

    def train(self):

        data = glob(os.path.join("../Datasets/celebA", "*.jpg"))
        base = np.array([get_image(sample_file, 108, is_crop=True) for sample_file in data[0:100]])
        base += 1
        base /= 2

        np.set_printoptions(threshold=np.inf)
        print base[0]

        ims("results/base.jpg",merge(base,[10,10]))
                # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(len(data) / self.batchsize):
                    # load the images
                    batch_files = data[idx*self.batchsize:(idx+1)*self.batchsize]
                    batch = [get_image(batch_file, 108, is_crop=True) for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
                    batch_images += 1
                    batch_images /= 2

                    _, gen_loss, lat_loss, imgs = sess.run((self.optimizer, self.generation_loss, self.latent_loss, self.generated_images_flat), feed_dict={self.images: batch_images})
                    print "iter %d: genloss %f latloss %f" % (epoch*10000 + idx, np.mean(gen_loss), np.mean(lat_loss))
                    print np.amin(imgs)
                    print np.amax(imgs)
                    if idx % 10 == 0:

                        # saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.cs, feed_dict={self.images: base})
                        for t in xrange(self.sequence_length):
                            ims("results/"+str(idx + epoch*10000)+"-"+str(t)+".jpg",merge(generated_test[t],[10,10]))


model = LatentAttention()
model.train()
