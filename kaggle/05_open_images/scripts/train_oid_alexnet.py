#! /usr/bin/env python

''' This program trains Alexnet on the OID challenge dataset '''

import matplotlib
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")

import os
import cv2
import numpy as np
import sys
import json
import logging
import progressbar

from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Local imports
import oi_utils
# from hdf5datasetwriter import HDF5DatasetWriter
# from aspectawarepreprocessor import AspectAwarePreprocessor
# from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import AlexNet

# Challenge datasets constants
BASE_DIR = '/data/fast1/'
DATASETS_BASE_DIR = os.path.join(BASE_DIR,'datasets/kaggle/open_images/' )

# Tiny datasets constants
TINY_DATASET_NAME = 'oid_tiny1'
TINY_DATASETS_BASE_DIR=os.path.join(BASE_DIR, 
                                'datasets/kaggle/open_images/tiny_datasets')
NEW_TINY_DATASET_DIR = os.path.join(TINY_DATASETS_BASE_DIR, TINY_DATASET_NAME)
TINY_TRAIN_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'train')
TINY_VAL_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'val')
TINY_TEST_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'test')

# HDF5 file specs
TINY_TRAIN_HDF5 = os.path.join(NEW_TINY_DATASET_DIR, 'hdf5/train.hdf5')
TINY_VAL_HDF5 = os.path.join(NEW_TINY_DATASET_DIR, 'hdf5/val.hdf5')
TINY_TEST_HDF5 = os.path.join(NEW_TINY_DATASET_DIR, 'hdf5/test.hdf5')

# Outputs
OUTPUT_PATH = './output'
MODEL_PATH = os.path.join(OUTPUT_PATH, 'alexnet_{}.model'
                                                 .format(TINY_DATASET_NAME))


# Other constants
TINY_RGB_MEAN_FILE = os.path.join(NEW_TINY_DATASET_DIR, '{}.json'
                                                 .format(TINY_DATASET_NAME))

# Set logging level
logging.basicConfig(level=logging.INFO)

def prep_for_training():
    ''' Prepare for training '''
    logging.info('Preparing for training ...')
    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
              width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                            horizontal_flip=True, fill_mode="nearest")

    # load the RGB means for the training set
    means = json.loads(open(TINY_RGB_MEAN_FILE).read())

    # initialize the image preprocessors
    sp = SimplePreprocessor(227, 227)
    pp = PatchPreprocessor(227, 227)
    mp = MeanPreprocessor(means["R"], means["G"], means["B"])
    iap = ImageToArrayPreprocessor()

    # initialize the training and validation dataset generators
    trainGen = HDF5DatasetGenerator(TINY_TRAIN_HDF5, 128, aug=aug,
                    preprocessors=[pp, mp, iap], classes=500)
    valGen = HDF5DatasetGenerator(TINY_VAL_HDF5, 128,
                    preprocessors=[sp, mp, iap], classes=500)

    # construct the set of callbacks
    path = os.path.sep.join([OUTPUT_PATH, "{}.png".format(os.getpid())])
    callbacks = [TrainingMonitor(path)]
    
    return trainGen, valGen, callbacks

def init_alex_model():
    ''' Initialize AlexNet model '''
    logging.info("Compiling model...")
    opt = Adam(lr=1e-3)
    model = AlexNet.build(width=227, height=227, depth=3,
                    classes=500, reg=0.0002)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                    metrics=["accuracy"])
    return model

def train_alexnet(alex_model, trainGen, valGen, callbacks):
    ''' Train Alexnet '''
    alex_model.fit_generator(
            trainGen.generator(),
            steps_per_epoch=trainGen.numImages // 128,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // 128,
            epochs=75,
            max_queue_size=10,
            callbacks=callbacks, verbose=1)

    # save the model to file
    logging.info("Serializing model...")
    alex_model.save(MODEL_PATH, overwrite=True)

    # close the HDF5 datasets
    trainGen.close()
    valGen.close()

def main():
    ''' Main program '''
    trainGen, valGen, callbacks = prep_for_training()
    alex_model = init_alex_model()
    # Now train the network
    train_alexnet(alex_model, trainGen, valGen, callbacks)


if __name__ == '__main__':
    main()
