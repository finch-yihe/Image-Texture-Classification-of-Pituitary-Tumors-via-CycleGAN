from __future__ import print_function, division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, AveragePooling2D, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import keras
from data_loader import DataLoader
import scipy.misc
import cv2

class FeatureExtraction():

    def __init__(self, channels, name):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.name = name
        optimizer = Adam(0.0002, 0.5)

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        z = Input(shape=(self.img_shape))
        self.feature = self.encoder(z)
        generator = self.decoder(self.feature)

        self.model = Model(z, generator)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    

    def build_encoder(self):

        def transition_conv(layer_input, filters):
            t = BatchNormalization(epsilon=1.001e-5)(layer_input)
            t = LeakyReLU(alpha=0.2)(t)
            t = Conv2D(filters, 1, use_bias=False)(t)
            t = AveragePooling2D(2, strides=2)(t)
            return t

        def conv_block(layer_input, growth_rate=4):
            c1 = BatchNormalization(epsilon=1.001e-5)(layer_input)
            c1 = LeakyReLU(alpha=0.2)(c1)
            c1 = Conv2D(4 * growth_rate, 1, use_bias=False)(c1)
            c1 = BatchNormalization(epsilon=1.001e-5)(c1)
            c1 = LeakyReLU(alpha=0.2)(c1)
            c1 = Conv2D(growth_rate, 3, padding='same',use_bias=False)(c1)
            c = Concatenate()([layer_input, c1])
            return c

        def dense_block(x, blocks=4):
            for i in range(blocks):
                x = conv_block(x)
            return x

        img = Input(shape=self.img_shape)

        d1 = dense_block(img)
        d2 = transition_conv(d1, 48)
        d3 = transition_conv(d2, 24)

        return Model(img, d3)

    def build_decoder(self):

        def transition_deconv(layer_input, filters):
            t = UpSampling2D(size=2)(layer_input)
            t = Conv2D(filters, 1, use_bias=False, activation='relu')(t)
            # equal keras.layers.deconv2d
            t = BatchNormalization(epsilon=1.001e-5)(t)
            return t

        def identity_block(layer_input, growth_rate=4):
            i1 = Conv2D(growth_rate, 1, use_bias=False)(layer_input)
            i1 = BatchNormalization(epsilon=1.001e-5)(i1)
            i1 = LeakyReLU(alpha=0.2)(i1)
            i1 = Conv2D(growth_rate, 3, padding='same', use_bias=False)(i1)
            i1 = BatchNormalization(epsilon=1.001e-5)(i1)
            i1 = LeakyReLU(alpha=0.2)(i1)
            i1 = Conv2D(4 * growth_rate, 1, use_bias=False)(i1)
            i1 = BatchNormalization(epsilon=1.001e-5)(i1)
            i1 = Add()([i1, layer_input])
            i1 = LeakyReLU(alpha=0.2)(i1)
            return i1


        def conv_block(layer_input, growth_rate=4):
            c1 = Conv2D(growth_rate, 1, use_bias=False)(layer_input)
            c1 = BatchNormalization(epsilon=1.001e-5)(c1)
            c1 = LeakyReLU(alpha=0.2)(c1)
            c1 = Conv2D(growth_rate, 3, padding='same', use_bias=False)(c1)
            c1 = BatchNormalization(epsilon=1.001e-5)(c1)
            c1 = LeakyReLU(alpha=0.2)(c1)
            c1 = Conv2D(4 * growth_rate, 1, use_bias=False)(c1)
            c1 = BatchNormalization(epsilon=1.001e-5)(c1)
            shortcut = Conv2D(4 * growth_rate, 1, use_bias=False)(layer_input)
            shortcut = BatchNormalization(epsilon=1.001e-5)(shortcut)
            c1 = Add()([c1, shortcut])
            c1 = LeakyReLU(alpha=0.2)(c1)
            return c1

        img = Input(shape=(int(self.img_rows/4), int(self.img_cols/4), self.channels))

        d1 = conv_block(img)
        d2 = identity_block(d1)
        d3 = identity_block(d2)
        d4 = identity_block(d3)
        d5 = transition_deconv(d4, 48)
        d6 = UpSampling2D(size=2)(d5)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(d6)

        return Model(img, output_img)
    
    def train(self, data, epochs, batch_size):
        os.makedirs('images/', exist_ok=True)
        for epoch in range(epochs):
            self.model.fit(data, data, epochs=1, batch_size=batch_size)
            out = self.model.predict(data[0,:,:,:].reshape(1, 256, 256, 24))
            # out = self.model.predict(data[0,:,:,:].reshape(1, 256, 256, 1))
            # gen_imgs = np.concatenate([data[0,:,:,6].reshape(1,256,256), cv2.resize(feature[0,:,:,6],(256,256)).reshape(1,256,256), out[0,:,:,0].reshape(1,256,256)])
            gen_imgs = np.concatenate([data[0,:,:,0].reshape(1,256,256),  out[0,:,:,0].reshape(1,256,256)])
            if epoch % 100 == 0:
                r = 2 
                titles = ['Original', 'Translated', 'Reconstructed']
                fig, axs = plt.subplots(r)
                cnt = 0
                for i in range(r):
                    axs[i].imshow(gen_imgs[cnt], cmap='gray')
                    axs[i].set_title(titles[i])
                    axs[i].axis('off')
                    cnt += 1
                fig.savefig("images/%s_%d.png" % (self.name, epoch))
                plt.close()
            if epoch % 100 == 0:
                self.encoder.save("model/{}_model_{}.model".format(self.name, epoch))

class FeatureExtraction1():

    def __init__(self, channels,name):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = channels
        self.name = name
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        optimizer = Adam(0.0002, 0.5)

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        z = Input(shape=(self.img_shape))
        self.feature = self.encoder(z)
        generator = self.decoder(self.feature)

        self.model = Model(z, generator)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    

    def build_encoder(self):

        def transition_conv(layer_input, filters):
            t = BatchNormalization(epsilon=1.001e-5)(layer_input)
            t = LeakyReLU(alpha=0.2)(t)
            t = Conv2D(filters, 1, use_bias=False)(t)
            t = AveragePooling2D(2, strides=2)(t)
            return t

        def conv_block(layer_input, growth_rate=4):
            c1 = BatchNormalization(epsilon=1.001e-5)(layer_input)
            c1 = LeakyReLU(alpha=0.2)(c1)
            c1 = Conv2D(4 * growth_rate, 1, use_bias=False)(c1)
            c1 = BatchNormalization(epsilon=1.001e-5)(c1)
            c1 = LeakyReLU(alpha=0.2)(c1)
            c1 = Conv2D(growth_rate, 3, padding='same',use_bias=False)(c1)
            c = Concatenate()([layer_input, c1])
            return c

        def dense_block(x, blocks=4):
            for i in range(blocks):
                x = conv_block(x)
            return x

        img = Input(shape=self.img_shape)

        d1 = dense_block(img)
        d2 = transition_conv(d1, 24)
        d3 = transition_conv(d2, 12)

        return Model(img, d3)

    def build_decoder(self):

        def transition_deconv(layer_input, filters):
            t = UpSampling2D(size=2)(layer_input)
            t = Conv2D(filters, 1, use_bias=False, activation='relu')(t)
            # equal keras.layers.deconv2d
            t = BatchNormalization(epsilon=1.001e-5)(t)
            return t

        def identity_block(layer_input, growth_rate=4):
            i1 = Conv2D(growth_rate, 1, use_bias=False)(layer_input)
            i1 = BatchNormalization(epsilon=1.001e-5)(i1)
            i1 = LeakyReLU(alpha=0.2)(i1)
            i1 = Conv2D(growth_rate, 3, padding='same', use_bias=False)(i1)
            i1 = BatchNormalization(epsilon=1.001e-5)(i1)
            i1 = LeakyReLU(alpha=0.2)(i1)
            i1 = Conv2D(4 * growth_rate, 1, use_bias=False)(i1)
            i1 = BatchNormalization(epsilon=1.001e-5)(i1)
            i1 = Add()([i1, layer_input])
            i1 = LeakyReLU(alpha=0.2)(i1)
            return i1


        def conv_block(layer_input, growth_rate=4):
            c1 = Conv2D(growth_rate, 1, use_bias=False)(layer_input)
            c1 = BatchNormalization(epsilon=1.001e-5)(c1)
            c1 = LeakyReLU(alpha=0.2)(c1)
            c1 = Conv2D(growth_rate, 3, padding='same', use_bias=False)(c1)
            c1 = BatchNormalization(epsilon=1.001e-5)(c1)
            c1 = LeakyReLU(alpha=0.2)(c1)
            c1 = Conv2D(4 * growth_rate, 1, use_bias=False)(c1)
            c1 = BatchNormalization(epsilon=1.001e-5)(c1)
            shortcut = Conv2D(4 * growth_rate, 1, use_bias=False)(layer_input)
            shortcut = BatchNormalization(epsilon=1.001e-5)(shortcut)
            c1 = Add()([c1, shortcut])
            c1 = LeakyReLU(alpha=0.2)(c1)
            return c1

        img = Input(shape=(int(self.img_rows/4), int(self.img_cols/4), self.channels))

        d1 = conv_block(img)
        d2 = identity_block(d1)
        d3 = identity_block(d2)
        d4 = identity_block(d3)
        d5 = transition_deconv(d4, 24)
        d6 = UpSampling2D(size=2)(d5)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(d6)

        return Model(img, output_img)
    
    def train(self, data, epochs, batch_size):
        os.makedirs('images/', exist_ok=True)
        for epoch in range(epochs):
            self.model.fit(data, data, epochs=1, batch_size=batch_size)
            out = self.model.predict(data[0,:,:,:].reshape(1, 256, 256, 12))
            # out = self.model.predict(data[0,:,:,:].reshape(1, 256, 256, 1))
            # gen_imgs = np.concatenate([data[0,:,:,6].reshape(1,256,256), cv2.resize(feature[0,:,:,6],(256,256)).reshape(1,256,256), out[0,:,:,0].reshape(1,256,256)])
            gen_imgs = np.concatenate([data[0,:,:,0].reshape(1,256,256),  out[0,:,:,0].reshape(1,256,256)])
            if epoch % 100 == 0:
                r = 2 
                titles = ['Original', 'Translated', 'Reconstructed']
                fig, axs = plt.subplots(r)
                cnt = 0
                for i in range(r):
                    axs[i].imshow(gen_imgs[cnt], cmap='gray')
                    axs[i].set_title(titles[i])
                    axs[i].axis('off')
                    cnt += 1
                fig.savefig("images/%s_%d.png" % (self.name, epoch))
                plt.close()
            if epoch % 100 == 0:
                self.encoder.save("model/{}_model_{}.model".format(self.name,epoch))

if __name__ == '__main__':

    instance = FeatureExtraction(24,"total")
    data = np.load(r'data/total_image_374.npy')
    instance.train(data, 501, 10)
    instance = FeatureExtraction1(12,"T1")
    data = np.load(r'data/T1_total_images_374.npy')
    instance.train(data, 501, 10)
    instance = FeatureExtraction1(12,"T2")
    data = np.load(r'data/T2_total_images_374.npy')
    instance.train(data, 501, 10)
    # plt.close()
    # feature = instance.encoder.predict(data[0,:,:,:].reshape(1, 256, 256, 6))
    # print(feature.shape)
    # plt.imshow(a)
    # plt.show()
    # T1model = instance.model
    # data = np.load(r'T1_images.npy')
    # print(np.min(data))
    # print(np.max(data))
    # data = np_normalize(data)
    # print(data.shape)
    # plt.imshow(data[0])
    # plt.show()
    # T1model.fit(data, data, epochs=1, batch_size=2)



