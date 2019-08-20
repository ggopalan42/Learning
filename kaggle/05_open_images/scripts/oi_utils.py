#! /usr/bin/env python

import os
import cv2
import sys
import logging
import argparse
import pickle

import numpy as np

from collections import OrderedDict, defaultdict

# Challenge datasets constants
BASE_DIR = '/data/fast1/'
DATASETS_BASE_DIR = os.path.join(BASE_DIR,'datasets/kaggle/open_images/' )
CSVS_BASE_DIR = os.path.join(BASE_DIR, 
                                   'datasets/kaggle/open_images/csv_files')
CHALLENGE2018_TRAIN_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_train')
CHALLENGE2018_VAL_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_val')
CHALLENGE2018_TEST_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_test')
CHALLENGE_CLASS_DESC_FILE = os.path.join(CSVS_BASE_DIR, 
                               'challenge-2018-class-desc-to-int-map.csv')
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

# Pickle file constants
BBOX_PICKLE = 'bbox.pkl'
CLASS_LABELS_PICKLE = 'classlabels.pkl'
IMAGE_LABELS_PICKLE = 'imagelabels.pkl'

# Set logging level
logging.basicConfig(level=logging.INFO)

class challenge_csv_info():
    ''' This obj holds information on the various challenege csvs '''
    def __init__(self, load_from_pickle = False):
        self.bbox_dict = defaultdict(lambda: [])
        self.imagelabels_dict = defaultdict(lambda: [])
        self.classlabels_dict = {}
        self.mr_labels_to_int_dict = {}    # mr = Machine Readable
        self.hr_labels_to_int_dict = {}    # hr = Human Readable
        self._convert_bbox_to_dict(load_from_pickle)
        # self._convert_imagelabels_to_dict(load_from_pickle)
        self._convert_classlabels_to_dict()

    # Private methods
    def _convert_bbox_to_dict(self, load_from_pickle):
        ''' Convert bbox csvs to dicts. 
        Otherwise processing them takes too long '''

        # Create the bbox_dict
        logging.info('Converting the entire bbox csv into a dict')
        if load_from_pickle:
            logging.info('Loading bbox dict from pickle file')
            with open(BBOX_PICKLE, 'rb') as rfh:
                self.bbox_dict = pickle.load(rfh)

        else:
            with open(CSV_BBOX_ANNO_FILE) as fh:
                for line in fh:
                    if line.startswith('ImageID'):
                        continue                  # Skip the header
                    else:
                        image_id = line.split(',')[0]
                        self.bbox_dict[image_id].append(line)
            # Convert to a ordinary dict so as to be compatible
            # when loading from pickle file. pickle file is stored as dict
            self.bbox_dict = dict(self.bbox_dict)

    def _convert_imagelabels_to_dict(self, load_from_pickle):
        ''' Convert imagelabels csvs to dicts. 
        Otherwise processing them takes too long '''
        logging.info('Converting the entire human-imagelabels csv into a dict')
        if load_from_pickle:
            logging.info('Loading imagelabels dict from pickle file')
            with open(IMAGE_LABELS_PICKLE, 'rb') as rfh:
                self.imagelabels_dict = pickle.load(rfh)
        else:
            with open(CSV_BBOX_IMAGELABELS_FILE) as fh:
                for line in fh:
                    if line.startswith('ImageID'):
                        continue                  # Skip the header
                    else:
                        image_id = line.split(',')[0]
                        self.imagelabels_dict[image_id].append(line)

            # Convert to a ordinary dict so as to be compatible
            # when loading from pickle file. pickle file is stored as dict
            self.imagelabels_dict = dict(self.imagelabels_dict)

    def _convert_classlabels_to_dict(self):
        ''' Convert and store the class labels in a dict '''
        logging.info('Converting class labels into a dict')
        with open(CHALLENGE_CLASS_DESC_FILE) as class_fh:
            for class_line in class_fh:
                class_split = class_line.strip().split(',')
                # Each line is for format: mr_label, hr_label, int
                # eg: /m/01kb5b,Flashlight,1003
                self.classlabels_dict[class_split[0]] = class_split[1]
                self.hr_labels_to_int_dict[class_split[1]] = np.int32(class_split[2])
                self.mr_labels_to_int_dict[class_split[0]] = np.int32(class_split[2])

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

    def get_hr_label_to_int(self, hr_label):
        ''' Given a Human Readable label (eg: Camera) return the 
            corresponding integer (eg: 1005) '''
        return self.hr_labels_to_int_dict[hr_label] 

    def get_mr_label_to_int(self, mr_label):
        ''' Given a Machine Readable label (eg: /m/0dv5r) return the 
            corresponding integer (eg: 1005) '''
        return self.mr_labels_to_int_dict[mr_label] 


if __name__ == '__main__':
    # Used for testing
    csv_info =  challenge_csv_info(load_from_pickle=True)

    # Test some of them out
    print(csv_info.get_class_label('/m/09f_2'))    # Should print Crocodile
    print(csv_info.get_hr_label_to_int('Vehicle registration plate'))   # Should print 1013
    print(csv_info.get_mr_label_to_int('/m/02hj4'))   # Should print 1487

