#! /usr/bin/env python
''' This program builds the hdf5 file for the real test set from the 
    competition. This is as opposed to the test hdf5 build by code
    from pyimagesearch that is really a split of the train set

    The hdf5 file for the competition test set is called: test_competition.hdf5
    The hdf5 file generated by original pyimagesearch code is called: test.hdf5
'''

# import the necessary packages
from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os


def get_test_paths():
    ''' This will return the filenames and paths of files in test set '''
    testPaths = list(paths.list_images(config.TEST_COMPETITION_PATH))
    return testPaths


def generate_test_hdf5(testPaths):
    ''' This will read each image from the competition test dir and generate
        a hdf5 file '''

    # initialize the image pre-processor
    aap = AspectAwarePreprocessor(256, 256)

    # create HDF5 writer
    print('[INFO] Building competition test hdf5...')
    writer = HDF5DatasetWriter((len(testPaths), 256, 256, 3),
                               config.TEST_COMPETITION_HDF5)

    # initialize the progress bar
    widgets = ["Building Competition Test HDF5: ", progressbar.Percentage(), 
               " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(testPaths),
                                   widgets=widgets).start()
    # loop over the test image paths
    for (i, path) in enumerate(testPaths):
        # load the image and process it
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # add the image and label # to the HDF5 dataset. And since this is 
        # test set, label is simply hardcoded to a 0
        writer.add([image], [0])  
        pbar.update(i)

    # close the HDF5 writer
    pbar.finish()
    writer.close()


if __name__ == '__main__':
    testPaths = get_test_paths()
    generate_test_hdf5(testPaths)