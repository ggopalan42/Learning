#! /usr/bin/env python

''' This program makes hdf5 files from tiny datasets (or any dataset for that
    matter) '''

import os
import cv2
import numpy as np
import sys
import json
import logging
import progressbar


# Local imports
import oi_utils
from hdf5datasetwriter import HDF5DatasetWriter
from aspectawarepreprocessor import AspectAwarePreprocessor

from collections import defaultdict

# Challenge datasets constants
BASE_DIR = '/data/fast1/'
TINY_DATASET_NAME = 'oid_tiny1'
DATASETS_BASE_DIR = os.path.join(BASE_DIR,'datasets/kaggle/open_images/' )

# Tiny datasets constants
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

# Other constants
TINY_RGB_MEAN_FILE = os.path.join(NEW_TINY_DATASET_DIR, 
                                       '{}.json'.format(TINY_DATASET_NAME))

# Set logging level
logging.basicConfig(level=logging.INFO)

def get_oi_datasets_info():
    ''' This will return a dict of train/val/test sets info '''

    # The dataset info will be of the format:
    #   ([img1 path, img2 path, ..], [img1_id, img2_id, ..], hdf5 full path)

    # Train info
    tiny_train_files = [os.path.join(TINY_TRAIN_DIR, x) 
                                          for x in os.listdir(TINY_TRAIN_DIR)]
    tiny_train_labels = [os.path.splitext(x)[0] 
                                          for x in os.listdir(TINY_TRAIN_DIR)]

    # val
    tiny_val_files = [os.path.join(TINY_VAL_DIR, x) 
                                          for x in os.listdir(TINY_VAL_DIR)]
    tiny_val_labels = [os.path.splitext(x)[0] 
                                          for x in os.listdir(TINY_VAL_DIR)]

    # Test
    tiny_test_files = [os.path.join(TINY_TEST_DIR, x) 
                                          for x in os.listdir(TINY_TEST_DIR)]
    tiny_test_labels = [os.path.splitext(x)[0] 
                                          for x in os.listdir(TINY_TEST_DIR)]

    dataset_info=[
               ("train", tiny_train_files, tiny_train_labels, TINY_TRAIN_HDF5),
               ("val", tiny_val_files, tiny_val_labels, TINY_VAL_HDF5),
               ("test", tiny_test_files, tiny_test_labels, TINY_TEST_HDF5)]
    return dataset_info

def get_denormalized_bbox(image, bbox_str):
    ''' From the image dimenstions and the normalized bbox string,
    return the denormalized rectangle co-ordinates '''
    # The bbox_str is of the format: Label, xmin, xmax, ymin, ymax
    bbox_list = bbox_str.split(',')
    label = bbox_list[0]
    (xmin, xmax, ymin, ymax)  = (float(bbox_list[1]), float(bbox_list[2]), 
                                      float(bbox_list[3]), float(bbox_list[4]))
    image_xmax = image.shape[1]
    image_ymax = image.shape[0]
    return (label, int(xmin * image_xmax), int(xmax * image_xmax), 
                                int(ymin * image_ymax), int(ymax * image_ymax))

def get_sub_images(image, bounding_boxes):
    ''' This will add the bbox rectangle and label to the image.
    It will further crop individual bbox images and return them '''

    sub_images = []
    for i, bbox_str in enumerate(bounding_boxes):
        # Get the labels and rectangle for each bbox 
        label, startX, endX, startY, endY =       \
        get_denormalized_bbox(image, bbox_str)
        # Now cut out the rectangle from the main image
        cut_image = image[startY:endY, startX:endX]
        # cut_image = image[startX:startY, endX:endY]
        # Now add (label, cut_image) to the set of sub_images
        sub_images.append((label, cut_image))
        # Return the annotated main image and the sub images
    return sub_images

def get_hdf5_file_dims(csv_info, datasets_info):
    ''' This will return the lengths of the number of sub-images per 
        train/val/test type '''
    
    hdf5_file_dims = defaultdict(int)

    for (dtype, in_files, image_ids, out_hdf5_fn) in datasets_info:
        for image_id in image_ids:
            # The number of bbox per image_id will be the number of
            # sub-images in that image. So add that to the hdf5_dims
            try:
                # Some image IDs do not exist in the csv_info
                hdf5_file_dims[dtype] += len(csv_info.get_bbox(image_id))
            except KeyError:
                # In that case, simply addf a 1 to the hdf5_dims
                hdf5_file_dims[dtype] += 1
    return hdf5_file_dims

def make_hdf5_datasets(csv_info, datasets_info):
    ''' Make the specified hdf5 dataset '''
    # Initialize the preprocessor and RGB mean tuple
    aap = AspectAwarePreprocessor(256, 256)
    (R, G, B) = ([], [], [])

    hdf5_num_images = get_hdf5_file_dims(csv_info, datasets_info)

    for (dtype, in_files, image_ids, out_hdf5_fn) in datasets_info:
        logging.info('Building {}...'.format(out_hdf5_fn))
        writer = HDF5DatasetWriter((hdf5_num_images[dtype], 256, 256, 3), 
                                                                out_hdf5_fn)

        # initialize the progress bar
        widgets = ["Building {} dataset: ".format(dtype), 
                            progressbar.Percentage(), " ", progressbar.Bar(), 
                            " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(in_files), 
                                                      widgets=widgets).start()

        # loop over the images
        for (i, (image_fn, image_id)) in enumerate(zip(in_files, image_ids)):
            # Read the image
            image = cv2.imread(image_fn)
            # If image is from train or val set (i.e. not test set)
            if not dtype == 'test':
                # Get the bounding box for that image
                try:
                   bbox_str = csv_info.get_bbox(image_id)
                except KeyError:
                    logging.warning('Could not find image ID {} in csv file. Continuing'.format(image_id))
                    continue
                # Get a list of labels and cut_images for that image
                sub_images = get_sub_images(image, bbox_str)
            # If image is from test set, some other procesisng needs to be done
            else:
                # For now, set entire test image as the cut image.
                # Later do object detection
                sub_images = [(image_id, image)]
            for (label, cut_image) in sub_images:
                # Preprocess the cut image to be of constant dims for the NN
                cut_image = aap.preprocess(cut_image)

                # if we are building the training dataset, then compute the
                # mean of each channel in the image, then update the
                # respective lists
                if dtype == "train":
                    (b, g, r) = cv2.mean(cut_image)[:3]
                    R.append(r)
                    G.append(g)
                    B.append(b)

                # add the image and label # to the HDF5 dataset
                pbar.update(i)
                # If image is from train or val set (i.e. not test set)
                if not dtype == 'test':
                    # HDF5 does not handle labels very well. Therefore use the
                    # HR label to into converted in oi_utils to comvert 
                    # label to an int
                    int_label = csv_info.get_hr_label_to_int(label)
                    writer.add([cut_image], [int_label])
                # If image is from test set, simple add that imahe and a
                # dummy label
                else:
                    writer.add([cut_image], [np.int32(2000)])

        # close the HDF5 writer
        writer.close()
        pbar.finish()

    logging.info('Serializing means . . .')
    rgb_means = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    with open(TINY_RGB_MEAN_FILE, "w") as fh:
        fh.write(json.dumps(rgb_means))


def main():
    ''' Main program '''
    datasets_info = get_oi_datasets_info()
    csv_info = oi_utils.challenge_csv_info(load_from_pickle = True)
    make_hdf5_datasets(csv_info, datasets_info)

if __name__ == '__main__':
    main()
