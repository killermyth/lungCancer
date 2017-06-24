#!/usr/bin/env python
# encoding: utf-8

"""
训练uNet网络用来切割疑似结节
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

img_rows = 512
img_cols = 512
smooth = 1.

train_npy_path = './train_npy/'
test_npy_path = './test_npy/'
train_imgs = 'lung_img_LKDS-00001.npy'
train_masks = 'nodule_mask_LKDS-00001.npy'
test_imgs = 'lung_img_LKDS-00012.npy'
test_masks = 'lung_mask_LKDS-00012.npy'
use_existing = True


class Unet(object):
    def __init__(self):
        print 123

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)

    def get_unet(self):
        inputs = Input((1, 512, 512))
        conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv1)

        conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv2)

        conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv3)

        conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv4)

        conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
        conv6 = Dropout(0.2)(conv6)
        conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
        conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
        conv7 = Dropout(0.2)(conv7)
        conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
        conv8 = Dropout(0.2)(conv8)
        conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
        conv9 = Dropout(0.2)(conv9)
        conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)
        model.summary()
        model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

        return model

    def train_predict(self):
        imgs_train = np.load(train_npy_path + train_imgs).astype(np.float32)
        imgs_mask_train = np.load(train_npy_path + train_masks).astype(np.float32)
        imgs_test = np.load(test_npy_path + test_imgs).astype(np.float32)
        # imgs_mask_test_true = np.load(test_masks).astype(np.float32)

        mean = np.mean(imgs_train)  # mean for data centering
        std = np.std(imgs_train)  # std for data normalization

        imgs_train -= mean  # images should already be standardized, but just in case
        imgs_train /= std

        print('-' * 30)
        print('Creating and compiling model...')
        print('-' * 30)
        model = self.get_unet()
        # Saving weights to unet.hdf5 at checkpoints
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
        #
        # Should we load existing weights?
        # Set argument for call to train_and_predict to true at end of script
        if use_existing:
            model.load_weights('./unet.hdf5')

        #
        # The final results for this tutorial were produced using a multi-GPU
        # machine using TitanX's.
        # For a home GPU computation benchmark, on my home set up with a GTX970
        # I was able to run 20 epochs with a training set size of 320 and
        # batch size of 2 in about an hour. I started getting reseasonable masks
        # after about 3 hours of training.
        #
        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=20, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])

        # loading best weights from training session
        print('-' * 30)
        print('Loading saved weights...')
        print('-' * 30)
        model.load_weights('./unet.hdf5')

        print('-' * 30)
        print('Predicting masks on test data...')
        print('-' * 30)
        num_test = len(imgs_test)
        imgs_mask_test = np.ndarray([num_test, 1, 512, 512], dtype=np.float32)
        for i in range(num_test):
            imgs_mask_test[i] = model.predict([imgs_test[i:i + 1]], verbose=0)[0]
        np.save('masksTestPredicted.npy', imgs_mask_test)
        # mean = 0.0
        # for i in range(num_test):
        #     mean += self.dice_coef_np(imgs_mask_test_true[i, 0], imgs_mask_test[i, 0])
        # mean /= num_test
        # print("Mean Dice Coeff : ", mean)


if __name__ == '__main__':
    unet = Unet()
    unet.train_predict()
