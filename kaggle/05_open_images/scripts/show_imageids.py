#! /usr/bin/env python

import os
import cv2
import sys
import logging
import argparse

from collections import OrderedDict, defaultdict

# Local imports
import oi_utils

# Challenge datasets constants
BASE_DIR = '/data/fast1/'
DATASETS_BASE_DIR = os.path.join(BASE_DIR,'datasets/kaggle/open_images/' )
CSVS_BASE_DIR = os.path.join(BASE_DIR, 
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

# Colors user for drawing in OpenV
COLORS = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
         }

# Set logging level
logging.basicConfig(level=logging.INFO)

def parse_args():
    ''' Parse the arguments and return a dict '''
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", default = DEFAULT_IMAGES,
                                    help="Comma seperated images list")
    args = vars(ap.parse_args())
    return args

def get_images_full_path(images_list):
    ''' Get the full fn of the list of images and which dataset 
                                                      they are part of '''
    images_dict = OrderedDict()
    for (dtype, dtype_path) in IMAGES_SEARCH_ORDER.items():
        logging.info('Searching images in {} set'.format(dtype))
        for image_tmp in images_list:
            # get image id regardless if images list has .jpg extension or not
            image_id = image_tmp.split('.')[0]
            # Check if image_id file exists
            image_full_fn = '{}.{}'.format(image_id, 'jpg')
            image_full_path = os.path.join(IMAGES_SEARCH_ORDER[dtype], 
                                                                image_full_fn)
            if os.path.isfile(image_full_path):
                images_dict[image_id] = {'from_set': dtype, 
                                                 'full_path': image_full_path}
    return images_dict

def get_bounding_boxes(csv_info, image_id, dtype):
    ''' Return a dict of bounding boxes and its class for specified image_id '''
    bounding_boxes = {}
    if dtype == 'test':
        logging.info('No bounding boxes for image from test set')
        return bounding_boxes
    else:
        bounding_boxes = csv_info.get_bbox(image_id)
        return bounding_boxes

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

def show_images_with_bbox(csv_info, images_dict):
    for (image_id, image_params) in images_dict.items():
        logging.info('showing image: {}'.format(image_id))
        # Format set type + image_id
        dtype = images_dict[image_id]['from_set']
        dtype_image_id = '{}-{}'.format(dtype, image_id)

        # Get the bounding boxes specified for each image and display it
        # returned bounding boxes are of the format:
        # Label, xmin, xmax, ymin, ymax
        bounding_boxes = get_bounding_boxes(csv_info, image_id, dtype)
        # bounding_boxes = ['Man,0.000000,0.339869,0.431250,0.975000', 'Mobile phone,0.360411,0.628385,0.348125,0.658750', 'Human face,0.130719,0.212885,0.510000,0.593750']

        # Read image and display it
        cv2.namedWindow(dtype_image_id)
        image_full_fn = images_dict[image_id]['full_path']
        image = cv2.imread(image_full_fn)
        for bbox_str in bounding_boxes:
            label, startX, endX, startY, endY =       \
                                      get_denormalized_bbox(image, bbox_str)
            cv2.rectangle(image, (startX, startY), (endX, endY), 
                                          COLORS['Green'], thickness=2)

            cv2.putText(image, label, (startX, startY-5), 
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                       color=COLORS['Green'], thickness=2)


        cv2.moveWindow(dtype_image_id, 500, 100)
        cv2.imshow(dtype_image_id, image)

        print('Press \'n\' for next image. Image will auto forward in 10 sec')
        if cv2.waitKey(10000) == ord('n'):
            continue

def main():
    ''' Main program '''
    # init Stuff
    args = parse_args()
    csv_info = oi_utils.challenge_csv_info()

    # Now process
    images_list = [x for x in args['images'].split(',')]
    images_list = [x.strip() for x in images_list]
    images_full_fn = get_images_full_path(images_list)
    if not images_full_fn:
        logging.error('Specified images do not exist in any of the tran/va/test'
                      ' directories. Noting to do. Exiting')
    else:
        show_images_with_bbox(csv_info, images_full_fn)

if __name__ == '__main__':
    main()
