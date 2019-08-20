#! /usr/bin/env python

''' This program creates pickle files of the bbox-annotations,
    image labels and class descriptions. This is so that these
    annotations can be read much quicker than directly from csv '''

import os
import cv2
import sys
import logging
import argparse
import pickle
import time

from collections import OrderedDict, defaultdict

# Challenge datasets constants
USER_HOME_DIR = '/home2/ggopalan/'
DATASETS_BASE_DIR = os.path.join(USER_HOME_DIR,'datasets/kaggle/open_images/' )
CSVS_BASE_DIR = os.path.join(USER_HOME_DIR, 
                                   'datasets/kaggle/open_images/csv_files')
CHALLENGE2018_TRAIN_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_train')
CHALLENGE2018_VAL_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_val')
CHALLENGE2018_TEST_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_test')
CHALLENGE_CLASS_DESC_FILE = os.path.join(CSVS_BASE_DIR, 
                               'challenge-2018-class-descriptions-500.csv')
DEFAULT_IMAGES = 'a153fd2d19c6b0e7'
# Strictly speaking, OrderedDict is not needed for below
IMAGES_SEARCH_ORDER = OrderedDict([('train', CHALLENGE2018_TRAIN_DIR), 
                       ('val', CHALLENGE2018_VAL_DIR), 
                       ('test', CHALLENGE2018_TEST_DIR)])
CSV_BBOX_ANNO_FILE = os.path.join(CSVS_BASE_DIR,
                                  'challenge-2018-train-annotations-bbox.csv')
CSV_BBOX_IMAGELABELS_FILE = os.path.join(CSVS_BASE_DIR,
                     'challenge-2018-train-annotations-human-imagelabels.csv')
CLASS_DESC_CSV_FILE = os.path.join(CSVS_BASE_DIR, 
                                  'challenge-2018-class-descriptions-500.csv')

# BBOX field index definitions
BBOX_IMAGEID_IDX = 0
BBOX_LABELNAME_IDX = 2
BBOX_CONFIDENCE_IDX = 3
BBOX_XMIN_IDX = 4
BBOX_XMAX_IDX = 5
BBOX_YMIN_IDX = 6
BBOX_YMAX_IDX = 7

# Set logging level
logging.basicConfig(level=logging.INFO)

class challenge_csv_info():
    ''' This obj holds information on the various challenege csvs '''
    def __init__(self):
        self.bbox_dict = defaultdict(lambda: [])
        self.imagelabels_dict = defaultdict(lambda: [])
        self.classlabels_dict = {}
        self._convert_bbox_to_dict()
        self._convert_imagelabels_to_dict()
        self._convert_classlabels_to_dict()

    # Private methods
    def _convert_bbox_to_dict(self):
        ''' Convert bbox csvs to dicts. 
        Otherwise processing them takes too long '''

        # Create the bbox_dict
        logging.info('Converting the entire bbox csv into a dict')
        with open(CSV_BBOX_ANNO_FILE) as fh:
            for line in fh:
                if line.startswith('ImageID'):
                    continue                  # Skip the header
                else:
                    image_id = line.split(',')[0]
                    self.bbox_dict[image_id].append(line)

    def _convert_imagelabels_to_dict(self):
        ''' Convert imagelabels csvs to dicts. 
        Otherwise processing them takes too long '''
        logging.info('Converting the entire human-imagelabels csv into a dict')
        with open(CSV_BBOX_IMAGELABELS_FILE) as fh:
            for line in fh:
                if line.startswith('ImageID'):
                    continue                  # Skip the header
                else:
                    image_id = line.split(',')[0]
                    self.imagelabels_dict[image_id].append(line)

    def _convert_classlabels_to_dict(self):
        ''' Convert and store the class labels in a dict '''
        logging.info('Converting class labels into a dict')
        with open(CHALLENGE_CLASS_DESC_FILE) as class_fh:
            for class_line in class_fh:
                class_split = class_line.strip().split(',')
                self.classlabels_dict[class_split[0]] = class_split[1]

    # Public methods
    def get_bbox(self, image_id):
        ''' Return the bounding box given the image id. This will
            also convert the class code to human readable code '''
        bbox_list = []
        for line in self.bbox_dict[image_id]:
            split_line = line.split(',')
            class_id = split_line[BBOX_LABELNAME_IDX]
            class_label = self.get_class_label(class_id)
            new_line = '{},{},{},{},{}'.format(
                  class_label, split_line[BBOX_XMIN_IDX],
                  split_line[BBOX_XMAX_IDX], split_line[BBOX_YMIN_IDX],
                  split_line[BBOX_YMAX_IDX])
            bbox_list.append(new_line)
        return(bbox_list)

    def get_class_label(self, class_id):
        ''' Given the class id (eg: /m/01gmv2) return the 
            class name (eg: Brassiere) '''
        return self.classlabels_dict[class_id] 


def save_pickle(obj, fn):
    with open(fn, 'wb') as wfh:
        pickle.dump(obj, wfh, pickle.HIGHEST_PROTOCOL)

def load_pickle(fn):
    with open(fn, 'rb') as rfh:
        obj = pickle.load(rfh) # , pickle.HIGHEST_PROTOCOL)
    return obj


if __name__ == '__main__':
    # Used for testing
    print('Loading from csv')
    oi_info = challenge_csv_info()
    print('Done loading ')

    print('Pickle bbox')
    save_pickle(dict(oi_info.bbox_dict), 'bbox.pkl')

    print('Pickle imagelabels')
    save_pickle(dict(oi_info.imagelabels_dict), 'imagelabels.pkl')

    print('Pickle classlabels')
    save_pickle(dict(oi_info.classlabels_dict), 'classlabels.pkl')

    print('Done all pickling')
