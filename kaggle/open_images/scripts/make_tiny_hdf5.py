#! /usr/bin/env python

import os
import cv2
import numpy as np
import sys
import json
import logging
import progressbar


# Local imports
from hdf5datasetwriter import HDF5DatasetWriter
from aspectawarepreprocessor import AspectAwarePreprocessor

from collections import defaultdict

# Challenge datasets constants
USER_HOME_DIR = '/home2/ggopalan/'
DATASETS_BASE_DIR = os.path.join(USER_HOME_DIR,'datasets/kaggle/open_images/' )

# Tiny datasets constants
TINY_DATASETS_BASE_DIR=os.path.join(USER_HOME_DIR, 
                                'datasets/kaggle/open_images/tiny_datasets')
NEW_TINY_DATASET_DIR = os.path.join(TINY_DATASETS_BASE_DIR, 'oid_tiny1')
TINY_TRAIN_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'train')
TINY_VAL_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'val')
TINY_TEST_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'test')

# HDF5 file specs
TINY_TRAIN_HDF5 = os.path.join(NEW_TINY_DATASET_DIR, 'hdf5/train.hdf5')
TINY_VAL_HDF5 = os.path.join(NEW_TINY_DATASET_DIR, 'hdf5/val.hdf5')
TINY_TEST_HDF5 = os.path.join(NEW_TINY_DATASET_DIR, 'hdf5/test.hdf5')

# Other constants
TINY_RGB_MEAN_FILE = os.path.join(NEW_TINY_DATASET_DIR, 'oid_tiny1.json')

# Set logging level
logging.basicConfig(level=logging.INFO)

def get_datasets_list():
    ''' Return a list with dataset type, path and other info in tuples '''
    # train
    tiny_train_files = [os.path.join(TINY_TRAIN_DIR, x) 
                                          for x in os.listdir(TINY_TRAIN_DIR)]
    tiny_train_labels = [os.path.splitext(x)[0] 
                                          for x in os.listdir(TINY_TRAIN_DIR)]
    tiny_train_labels = [np.int32(10) for x in tiny_train_labels]

    # val
    tiny_val_files = [os.path.join(TINY_VAL_DIR, x) 
                                          for x in os.listdir(TINY_VAL_DIR)]
    tiny_val_labels = [os.path.splitext(x)[0] 
                                          for x in os.listdir(TINY_VAL_DIR)]
    tiny_val_labels = [np.int32(10) for x in tiny_val_labels]

    # Test
    tiny_test_files = [os.path.join(TINY_TEST_DIR, x) 
                                          for x in os.listdir(TINY_TEST_DIR)]
    tiny_test_labels = [os.path.splitext(x)[0] 
                                          for x in os.listdir(TINY_TEST_DIR)]
    tiny_test_labels = [np.int32(10) for x in tiny_test_labels]

    datasets=[
               ("train", tiny_train_files, tiny_train_labels, TINY_TRAIN_HDF5),
               ("val", tiny_val_files, tiny_val_labels, TINY_VAL_HDF5),
               ("test", tiny_test_files, tiny_test_labels, TINY_TEST_HDF5)]
    return datasets

def make_hdf5_datasets(datasets):
    ''' Make the specified hdf5 dataset '''
    # Initialize the preprocessor and RGB mean tuple
    aap = AspectAwarePreprocessor(256, 256)
    (R, G, B) = ([], [], [])

    for (dtype, in_files, in_labels, out_hdf5_fn) in datasets:
        logging.info('Building {}...'.format(out_hdf5_fn))
        writer = HDF5DatasetWriter((len(in_files), 256, 256, 3), out_hdf5_fn)


        # initialize the progress bar
        widgets = ["Building {} dataset: ".format(dtype), 
                            progressbar.Percentage(), " ", progressbar.Bar(), 
                            " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(in_files), 
                                                      widgets=widgets).start()


        # loop over the images
        for (i, (image_fn, label)) in enumerate(zip(in_files, in_labels)):
            # load the image and process it
            image = cv2.imread(image_fn)
            image = aap.preprocess(image)

            # if we are building the training dataset, then compute the
            # mean of each channel in the image, then update the
            # respective lists
            if dtype == "train":
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)

            # add the image and label # to the HDF5 dataset
            pbar.update(i)
            writer.add([image], [label])


        # close the HDF5 writer
        writer.close()
        pbar.finish()

    logging.info('Serializing means . . .')
    rgb_means = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    with open(TINY_RGB_MEAN_FILE, "w") as fh:
        fh.write(json.dumps(rgb_means))



def main():
    ''' Main program '''
    datasets = get_datasets_list()
    make_hdf5_datasets(datasets)

if __name__ == '__main__':
    main()
