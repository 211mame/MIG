
import os
import argparse

import numpy as np

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD

import models

def normalization(X):
    return X / 127.5 - 1
def inverse_normalization(X):
    return (X + 1.) / 2.

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def load_data(dataset_path):
    #データセット読み込み用関数
    pass


def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1
    else:
        X_disc = procImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc


def train(args):
    if not os.path.exists(os.path.expanduser(args.datasetpath)):
        os.mkdir(findername)
    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    procImage, rawImage, procImage_val, rawImage_val = load_data(args.datasetpath)

    img_shape = rawImage.shape[-3:]
    patch_num = (img_shape[0] // args.patch_size) * (img_shape[1] // args.patch_size)
    disc_img_shape = (args.patch_size, args.patch_size, procImage.shape[-1])

    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    generator_model = models.load_generator(img_shape, disc_img_shape)
    discriminator_model = models.load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)

    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = models.load_DCGAN(generator_model, discriminator_model, img_shape, args.patch_size)

    loss = [l1_loss, 'binary_crossentropy']
    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    print('start')
    for e in range(args.epoch):
        perm = np.random.permutation(rawImage.shape[0])
        X_procImage = procImage[perm]
        X_rawImage  = rawImage[perm]
        X_procImageIter = [X_procImage[i:i+args.batch_size] for i in range(0, rawImage.shape[0], args.batch_size)]
        X_rawImageIter  = [X_rawImage[i:i+args.batch_size] for i in range(0, rawImage.shape[0], args.batch_size)]
        b_it = 0
        progbar = generic_utils.Progbar(len(X_procImageIter)*args.batch_size)
        for (X_proc_batch, X_raw_batch) in zip(X_procImageIter, X_rawImageIter):
            b_it += 1
            X_disc, y_disc = get_disc_batch(X_proc_batch, X_raw_batch, generator_model, b_it, args.patch_size)
            raw_disc, _ = get_disc_batch(X_raw_batch, X_raw_batch, generator_model, 1, args.patch_size)
            x_disc = X_disc + raw_disc
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

            idx = np.random.choice(procImage.shape[0], args.batch_size)
            X_gen_target, X_gen = procImage[idx], rawImage[idx]
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])

            discriminator_model.trainable = True

            progbar.add(args.batch_size, values=[
                ("D logloss", disc_loss),
                ("G tot", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

            if b_it % (procImage.shape[0]//args.batch_size//2) == 0:
                idx = np.random.choice(procImage_val.shape[0], args.batch_size)
                X_gen_target, X_gen = procImage_val[idx], rawImage_val[idx]

def main():
    parser = argparse.ArgumentParser(description='Train Font GAN')
    parser.add_argument('--dataset_path', '-d', type=str, required=True)
    parser.add_argument('--patch_size', '-p', type=int, default=64)
    parser.add_argument('--batch_size', '-b', type=int, default=5)
    parser.add_argument('--epoch','-e', type=int, default=400)
    args = parser.parse_args()

    K.set_image_data_format("channels_last")

    train(args)


if __name__=='__main__':
    main()
