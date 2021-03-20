from __future__ import print_function, division

from keras.backend import categorical_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import datetime
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import sys
import keras
import numpy as np
import time


def mnist_cnn1(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss=categorical_crossentropy, optimizer=sgd,
                  metrics=['accuracy'])
    return model


class mnist_GAN():
    def __init__(self, input_shape, input_latent_dim, G_data, D_data, image_path):
        """

        :param input_shape:
        :param input_latent_dim: the shape input noise of G,should be 1-D array
        :param datasets: the datasets,should be numpy array
        :param image_path: image save path during training
        """
        self.img_shape = input_shape
        self.latent_dim = input_latent_dim
        self.G_datasets = G_data
        self.D_datasets = D_data
        self.image_path = image_path
        self.log = []
        optimizer = Adam(0.00001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        frozen_D = Model(
            inputs=self.discriminator.inputs,
            outputs=self.discriminator.outputs)
        frozen_D.trainable = False
        reconstructed_z = self.generator(z)
        validity = frozen_D(reconstructed_z)
        # The discriminator takes generated images as input and determines validity

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        # self.combined = Model(z, validity)
        # self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.combined = Model(z, [reconstructed_z, validity])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Dense(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=(self.img_shape,))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50, rescale=False, expand_dims=True):
        """
        :param epochs: the iteration of training
        :param batch_size: batch_size
        :param sample_interval: print the loss of G and D each sample_interval
        :param rescale: if true,rescale D_img to [-1,1]
        :param expand_dims: if true,expand img channel ,for mnist [28,28]->[28,28,1] it's necessary
        :return:
        """

        # Load the dataset
        D_train = self.D_datasets
        G_train = self.G_datasets

        if rescale:
            # Rescale -1 to 1
            D_train = D_train / 127.5 - 1.
        if expand_dims:
            D_train = np.expand_dims(D_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, D_train.shape[0], batch_size)
            D_imgs = D_train[idx]  # targeted feature
            G_feature = G_train[idx]  # input feature

            noise_add = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = G_feature  # + noise_add

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(D_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, [D_imgs, valid])
            # If at save interval => save generated image samples

            if epoch % sample_interval == 0:
                # Plot the progress
                # message = "%d D loss: %.4f, acc.: %.2f%% G loss: %.4f mse:%.4f r2:%.4f" \
                #           % (epoch, d_loss[0], 100 * d_loss[1], g_loss, mse, r2)
                # self.log.append([epoch, d_loss[0], d_loss[1], g_loss])
                message = "%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1])
                self.log.append([epoch, d_loss[0], d_loss[1], g_loss[0], g_loss[1]])
                self.create_str_to_txt('cnn1', datetime.datetime.now().strftime('%Y-%m-%d'), message)
                print(message)
                # self.sample_images(epoch)

    def showlogs(self, path):
        logs = np.array(self.log)
        names = ["d_loss", "d_acc", "g_loss", "g_mse"]
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.plot(logs[:, 0], logs[:, i + 1])
            plt.xlabel("iteration")
            plt.ylabel(names[i])
            plt.grid()
        plt.tight_layout()
        plt.savefig(path+".png")
        plt.close()
        np.save(path+".npy",logs)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        if not os.path.isdir(self.image_path):
            os.makedirs(self.image_path)
        fig.savefig(self.image_path + "/%d.png" % epoch)
        plt.close()

    def save_model(self, path):
        self.combined.save(path)

    def load_model(self, path):
        self.combined.load_weights(path)

    def get_generator(self):
        return self.generator

    def calculateMSE(self, Y, Y_hat):
        MSE = np.sum(np.power((Y - Y_hat), 2)) / len(Y)
        R2 = 1 - MSE / np.var(Y)
        return MSE, R2

    def create_str_to_txt(self, model_name, date, str_data):
        """
        创建txt，并且写入
        """
        path_file_name = './adv_mnist/{}/mnist_{}_gan_{}.txt'.format(model_name, model_name, date)
        if not os.path.exists(path_file_name):
            with open(path_file_name, "w") as f:
                print(f)

        with open(path_file_name, "a") as f:
            f.write(str_data + '\n')


class mnist_p2f_GAN():
    def __init__(self, input_shape, input_latent_dim, G_data, D_data, image_path):
        """

        :param input_shape:
        :param input_latent_dim: the shape input noise of G,should be 1-D array
        :param datasets: the datasets,should be numpy array
        :param image_path: image save path during training
        """
        self.img_shape = input_shape
        self.latent_dim = input_latent_dim
        self.G_datasets = G_data
        self.D_datasets = D_data
        self.image_path = image_path
        self.log = []
        optimizer = Adam(0.00001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        frozen_D = Model(
            inputs=self.discriminator.inputs,
            outputs=self.discriminator.outputs)
        frozen_D.trainable = False
        reconstructed_z = self.generator(z)
        validity = frozen_D(reconstructed_z)
        # The discriminator takes generated images as input and determines validity

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        # self.combined = Model(z, [reconstructed_z, validity])
        # self.combined.compile(loss=['mse', 'binary_crossentropy'],
        #                       loss_weights=[0.999, 0.001],
        #                       optimizer=optimizer)

        # self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        # self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
        #                                      loss_weights=[0.999, 0.001],
        #                                      optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Dense(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=(self.img_shape,))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50, rescale=False, expand_dims=True):
        """
        :param epochs: the iteration of training
        :param batch_size: batch_size
        :param sample_interval: print the loss of G and D each sample_interval
        :param rescale: if true,rescale D_img to [-1,1]
        :param expand_dims: if true,expand img channel ,for mnist [28,28]->[28,28,1] it's necessary
        :return:
        """

        # Load the dataset
        D_train = self.D_datasets
        G_train = self.G_datasets

        if rescale:
            # Rescale -1 to 1
            D_train = D_train / 127.5 - 1.
        if expand_dims:
            D_train = np.expand_dims(D_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, D_train.shape[0], batch_size)
            D_imgs = D_train[idx]
            G_feature = G_train[idx]

            noise_add = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = G_feature + noise_add

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(D_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise_add = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = G_feature + noise_add
            mse, r2 = self.calculateMSE(D_imgs, gen_imgs)
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Plot the progress
                message = "%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1])
                self.log.append([epoch, d_loss[0], d_loss[1], g_loss[0], g_loss[1]])
                self.create_str_to_txt('cnn1', datetime.datetime.now().strftime('%Y-%m-%d'), message)
                print(message)
                # self.sample_images(epoch)

    def showlogs(self, path):
        logs = np.array(self.log)
        names = ["d_loss", "d_acc", "g_loss", "g_mse"]
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.plot(logs[:, 0], logs[:, i + 1])
            plt.xlabel("epoch")
            plt.ylabel(names[i])
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        if not os.path.isdir(self.image_path):
            os.makedirs(self.image_path)
        fig.savefig(self.image_path + "/%d.png" % epoch)
        plt.close()

    def save_model(self, path):
        self.combined.save(path)

    def load_model(self, path):
        self.combined.load_weights(path)

    def get_generator(self):
        return self.generator

    def calculateMSE(self, Y, Y_hat):
        MSE = np.sum(np.power((Y - Y_hat), 2)) / len(Y)
        R2 = 1 - MSE / np.var(Y)
        return MSE, R2

    def create_str_to_txt(self, model_name, date, str_data):
        """
        创建txt，并且写入
        """
        path_file_name = './adv_mnist/{}/mnist_{}_gan_{}.txt'.format(model_name, model_name, date)
        if not os.path.exists(path_file_name):
            with open(path_file_name, "w") as f:
                print(f)

        with open(path_file_name, "a") as f:
            f.write(str_data + '\n')


def get_sub_model(start_layer_name):
    """
    :param start_layer_name:
    :return: return a sub_model start with the start_layer's input
    """
    start_name = start_layer_name
    new_input = keras.layers.Input(batch_shape=model.get_layer(name=start_layer_name).get_input_shape_at(0))
    print(model.get_layer(name=start_layer_name).get_input_shape_at(0))
    layers_list = [layer.name for layer in model.layers]

    for index, name in enumerate(layers_list):
        if name == start_name:
            sub_list = layers_list[index:]
            break

    for index, sub_layer in enumerate(sub_list):
        if index == 0:
            new_output = model.get_layer(sub_layer)(new_input)
        else:
            new_output = model.get_layer(sub_layer)(new_output)

    sub_model = keras.Model(inputs=new_input, outputs=new_output)
    print(f"Sub_model {sub_list[0]} to {sub_list[-1]}")
    print(f"Sub_model's input is {new_input} and the output is {new_output}")
    return new_input, new_output, sub_model


def scheduler(epoch):
    if epoch <= 80:
        return 0.01
    if epoch <= 140:
        return 0.005
    return 0.001


if __name__ == '__main__':
    from keras.datasets import mnist
    import os
    import keras.backend.tensorflow_backend as K
    import tensorflow as tf

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    attack_list = ['BIM','MIFGSM','JSMA','PWA','LSA','CRA']
    attack_name = attack_list[0]
    adv_data_path = ''
    adv_label_path = ''
    adv_test_data_path = ''
    adv_test_label_path = ''

    train_SF_path = ''
    train_SF_adv_path = ''
    train_NSF_path = ''
    train_NSF_adv_path = ''

    test_SF_path = ''
    test_SF_adv_path = ''
    test_NSF_path = ''
    test_NSF_adv_path = ''

    SF_GAN_log_path = ''
    NSF_GAN_log_path = ''
    detector_save_path = ''
    detector_history_path = ''
    SF_gan_save_path = ''
    NSF_gan_save_path = ''

    train_X, train_y = mnist.load_data()[0]
    train_X = train_X.reshape(-1, 28, 28, 1)
    train_X = train_X.astype('float32')
    train_X /= 255
    test_X, test_y = mnist.load_data()[1]
    test_X = test_X.reshape(-1, 28, 28, 1)
    test_X = test_X.astype('float32')
    test_X /= 255
    x_train, x_validation = train_X / 255., test_X / 255.
    train_X = train_X[0:10000]
    train_y = train_y[0:10000]
    test_X = test_X[0:2000]
    test_y = test_y[0:2000]
    print(np.shape(train_X), np.shape(train_y), np.shape(test_X), np.shape(test_y))

    model = mnist_cnn1(input_shape=train_X.shape[1:])
    # model.summary()
    model.load_weights(".mnist_cnn1.h5")
    adv_data = np.load(adv_data_path)
    print(np.shape(adv_data))
    adv_data_y = np.load(adv_label_path)
    adv_data_y = to_categorical(adv_data_y, 10)

    adv_test_data = np.load(adv_test_data_path)
    adv_test_data_y = np.load(adv_test_label_path)
    print(np.shape(adv_test_data))
    adv_test_data_y = to_categorical(adv_test_data_y, 10)
    # exit(0)

    loss, accuracy = model.evaluate(adv_data, adv_data_y, verbose=2)
    print('adv  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    dense1_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
    dense1_layer_model.summary()
    print(dense1_layer_model.output)
    block = dense1_layer_model.predict(train_X[0:10000], batch_size=64)
    block_adv = dense1_layer_model.predict(adv_data[0:10000], batch_size=64)
    np.save('',block)
    np.save('',block_adv)
    np.save('',train_y)
    print(np.shape(block))
    print(np.shape(block_adv))
    print(np.shape(train_y))
    exit(0)
    test_block = dense1_layer_model.predict(test_X, batch_size=64)
    test_block_adv = dense1_layer_model.predict(adv_test_data, batch_size=64)

    gan_epochs = 25001
    gan_batchsize = 64

    # train SF_model
    targeted_feature = np.concatenate((block, block))
    input_block = np.concatenate((block, block_adv))
    print(np.shape(input_block), np.shape(targeted_feature))
    print("\n" * 5)
    print("training SF_model")
    time_start = time.time()
    gan = mnist_GAN(input_shape=128, input_latent_dim=128, G_data=input_block, D_data=targeted_feature,
                    image_path='./f2f/SF')
    gan.train(epochs=gan_epochs, batch_size=gan_batchsize, sample_interval=200, rescale=False, expand_dims=False)
    gan.showlogs(path=SF_GAN_log_path)
    model_save_path = ""
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    gan.save_model(model_save_path + SF_gan_save_path)
    train_SF_pre = gan.generator.predict(block, batch_size=64)
    train_SF_adv_pre = gan.generator.predict(block_adv, batch_size=64)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    np.save(train_SF_path, train_SF_pre)
    np.save(train_SF_adv_path, train_SF_adv_pre)

    test_SF_pre = gan.generator.predict(test_block, batch_size=64)
    test_SF_adv_pre = gan.generator.predict(test_block_adv, batch_size=64)
    np.save(test_SF_path, test_SF_pre)
    np.save(test_SF_adv_path, test_SF_adv_pre)


    print("\n" * 5)
    print("training NSF_model")
    # train NSF_model
    targeted_feature = np.concatenate((block_adv, block_adv))
    input_block = np.concatenate((block, block_adv))
    print(np.shape(input_block), np.shape(targeted_feature))

    gan = mnist_GAN(input_shape=128, input_latent_dim=128, G_data=input_block, D_data=targeted_feature,
                    image_path='./f2f/NSF')
    gan.train(epochs=gan_epochs, batch_size=gan_batchsize, sample_interval=200, rescale=False, expand_dims=False)
    gan.showlogs(path=NSF_GAN_log_path)
    model_save_path = ""
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    gan.save_model(model_save_path + NSF_gan_save_path)
    train_NSF_pre = gan.generator.predict(block, batch_size=64)
    train_NSF_adv_pre = gan.generator.predict(block_adv, batch_size=64)
    np.save(train_NSF_path, train_NSF_pre)
    np.save(train_NSF_adv_path, train_NSF_adv_pre)

    test_NSF_pre = gan.generator.predict(test_block, batch_size=64)
    test_NSF_adv_pre = gan.generator.predict(test_block_adv, batch_size=64)
    np.save(test_NSF_path, test_NSF_pre)
    np.save(test_NSF_adv_path, test_NSF_adv_pre)

    # testing the acc based on ori_model
    new_input, new_output, sub_model = get_sub_model('dropout_2')
    sub_model.summary()
    sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    sub_model.compile(loss=categorical_crossentropy, optimizer=sgd,
                      metrics=['accuracy'])
    print(sub_model.input)
    print("-"*5, "evaluate train data","-"*5)
    loss, accuracy = sub_model.evaluate(block, adv_data_y, verbose=2)
    print('SF train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(train_SF_pre, adv_data_y, verbose=2)
    print('SF pre_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(block_adv, adv_data_y, verbose=2)
    print('SF adv_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(train_SF_adv_pre, adv_data_y, verbose=2)
    print('SF pre_adv_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))

    loss, accuracy = sub_model.evaluate(block, adv_data_y, verbose=2)
    print('NSF train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(train_NSF_pre, adv_data_y, verbose=2)
    print('NSF pre_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(block_adv, adv_data_y, verbose=2)
    print('NSF adv_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(train_NSF_adv_pre, adv_data_y, verbose=2)
    print('NSF pre_adv_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))

    print("-" * 5, "evaluate test data", "-" * 5)
    loss, accuracy = sub_model.evaluate(test_block, adv_test_data_y, verbose=2)
    print('SF test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(test_SF_pre, adv_test_data_y, verbose=2)
    print('SF pre_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(test_block_adv, adv_test_data_y, verbose=2)
    print('SF adv_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(test_SF_adv_pre, adv_test_data_y, verbose=2)
    print('SF pre_adv_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))

    loss, accuracy = sub_model.evaluate(test_block, adv_test_data_y, verbose=2)
    print('NSF test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(test_NSF_pre, adv_test_data_y, verbose=2)
    print('NSF pre_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(test_block_adv, adv_test_data_y, verbose=2)
    print('NSF adv_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
    loss, accuracy = sub_model.evaluate(test_NSF_adv_pre, adv_test_data_y, verbose=2)
    print('NSF pre_adv_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))

    exit(0)


