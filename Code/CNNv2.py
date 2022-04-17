
from __future__ import absolute_import
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import cv2
import random
from tqdm import tqdm
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
BATCH_SIZE = 1
MAX_LENGTH = 1000
EPOCH = 25
'''
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
'''

def ssim_loss(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def import_data(path, IMG_SIZE):
    data = []
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            data.append([new_array])  # add this to our training_data
        except Exception as Ex:
            print(Ex)
    random.shuffle(data)
    return data


datasets = [
    {
        'name':'CT Brain Scans',
        'training_path':'C:\\Users\\charl\\Documents\\College\\Spring 2022\\Thesis\\Data\\CT and MRI brain scans\\Dataset\\images\\trainA',
        'np_training_path':'Numpy\\Training-CT-Brain-Scans.npy',
        'test_path':'C:\\Users\\charl\\Documents\\College\\Spring 2022\\Thesis\\Data\\CT and MRI brain scans\\Dataset\\images\\testA',
        'np_test_path':'Numpy\\Training-CT-Brain-Scans.npy',
        'IMG_SIZE':512
    },
    {
        'name':'MRI Brain Scans',
        'training_path':'C:\\Users\\charl\\Documents\\College\\Spring 2022\\Thesis\\Data\\CT and MRI brain scans\\Dataset\\images\\trainB',
        'np_training_path':'Numpy\\Training-MRI-Brain-Scans.npy',
        'test_path':'C:\\Users\\charl\\Documents\\College\\Spring 2022\\Thesis\\Data\\CT and MRI brain scans\\Dataset\\images\\testB',
        'np_test_path':'Numpy\\Training-MRI-Brain-Scans.npy',
        'IMG_SIZE':256
    },
]

for data_dict in datasets:
    #Load Data
    try:
        x_train = np.load(data_dict['np_training_path'])
        x_test = np.load(data_dict['np_test_path'])

    except:
        x_train = import_data(data_dict['training_path'], data_dict['IMG_SIZE'])
        x_test = import_data(data_dict['test_path'], data_dict['IMG_SIZE'])
        np.save(data_dict['np_training_path'], x_train)
        np.save(data_dict['np_test_path'], x_test)


    #Shape
    x_train = np.array(x_train).reshape(-1, data_dict['IMG_SIZE'], data_dict['IMG_SIZE'], 1)
    x_test =np.array(x_test).reshape(-1, data_dict['IMG_SIZE'], data_dict['IMG_SIZE'], 1)

    # #Normalize
    x_train = x_train / 255.
    x_test = x_test / 255.
    
    input_img = keras.Input(shape=(data_dict['IMG_SIZE'], data_dict['IMG_SIZE'], 1))
    x = keras.layers.Conv2D(32, (5, 5), (2,2), activation='tanh', padding='same') (input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (5, 5), (2,2), activation='tanh', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(8, (5, 5), (2,2), activation='tanh', padding='same')(x)
    encoded = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2DTranspose(8, (5, 5), (2,2), activation='tanh', padding='same')(encoded)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2DTranspose(64, (5, 5), (2,2), activation='tanh', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2DTranspose(32, (5, 5), (2,2), activation='tanh', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    decoded = keras.layers.Conv2DTranspose(1, (5, 5), activation='sigmoid', padding='same')(x)
    

    autoencoder = keras.Model(input_img, decoded)
    
    # lr = 0.001
    lr = tf.keras.optimizers.schedules.ExponentialDecay(0.005, 
                                                        decay_steps=100000, 
                                                        decay_rate=0.95)
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    loss=[keras.losses.BinaryCrossentropy(),keras.losses.MeanSquaredError(),ssim_loss]
    loss_weights= [.5, .25, .25]
    autoencoder.compile(optimizer=opt, loss=loss, loss_weights=loss_weights, metrics=['binary_crossentropy','mse',ssim_loss, psnr])
    autoencoder.summary()
    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ModelCheckpoint("Model\\" + data_dict['name'] + ' CNNv2.model', save_freq='epoch')]

    autoencoder.fit(x_train[:MAX_LENGTH], x_train[:MAX_LENGTH],
                    epochs=EPOCH,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=callbacks)

    autoencoder.save("Model\\" + data_dict['name'] + ' CNNv2.model')