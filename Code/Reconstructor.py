
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
import random
import imageio
from PIL import Image as im
def ssim_loss(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
BATCH_SIZE = 1
IMG_SIZE = 512
MAX_LENGTH = 32

def import_data(path):
    data = []
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            data.append([new_array])  # add this to our training_data
        except Exception as Ex:
            print(Ex)
    return data

if 'yes' in input('User Input:').lower():
    model_path = input('Model Path:')
    test_data_path = input('Test Data Path:')
else:
    model_path = 'C:\\Users\\charl\\Documents\\College\\Spring 2022\\Thesis\\Code\\Model\\Brain Classified Reconstruction Test CNNv4.model'
    test_data_path = 'C:\\Users\\charl\\Documents\\College\\Spring 2022\\Thesis\\Data\\Hemorrage\\Test_Collection\\Norm'

autoencoder = keras.models.load_model(model_path, custom_objects={'ssim_loss':ssim_loss, 'psnr':psnr})
autoencoder.summary()
x_test = import_data(test_data_path)
x_test =np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = x_test / 255.

decoded_imgs = autoencoder.predict(x_test[:MAX_LENGTH])
x = 15
for img in decoded_imgs:
    imageio.imwrite(f"Outputs\\{x}.png", img)
    x += 1