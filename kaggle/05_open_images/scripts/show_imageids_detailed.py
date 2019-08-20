#! /usr/bin/env python

import os
import cv2
import sys
import logging
import argparse

from collections import OrderedDict, defaultdict

# Local imports
import oi_utils
from aspectawarepreprocessor import AspectAwarePreprocessor

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

# Image constants
IMAGE_SHOW_WINDOW_X = 500
IMAGE_SHOW_WINDOW_Y = 100
MAX_SHOW_SUB_IMG = 20

SHOW_PREPROCESS = True


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

def get_main_and_sub_images(image, bounding_boxes): 
    ''' This will add the bbox rectangle and label to the image.
        It will further crop individual bbox images and return them '''

    sub_images = {}
    image_copy = image.copy()
    for i, bbox_str in enumerate(bounding_boxes):
        # Get the labels and rectangle for each bbox 
        label, startX, endX, startY, endY =       \
                                  get_denormalized_bbox(image, bbox_str)
        # Annotate the main image
        cv2.rectangle(image_copy, (startX, startY), (endX, endY), 
                                      COLORS['Green'], thickness=2)

        cv2.putText(image_copy, label, (startX, startY-5), 
                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                   color=COLORS['Green'], thickness=2)

        # Now cut out the rectangle from the main image
        sub_image_name = '{} (subimg-{})'.format(label, i) 
        cut_image = image[startY:endY, startX:endX]
        # cut_image = image[startX:startY, endX:endY]
        sub_images[sub_image_name] = cut_image

    # Return the annotated main image and the sub images
    return image_copy, sub_images

def show_main_and_sub_images(dtype_image_id, image, sub_images):
    ''' Show the annotated main and sub-images '''
    aap = AspectAwarePreprocessor(256, 256)
    # Show main image
    cv2.namedWindow(dtype_image_id)
    cv2.moveWindow(dtype_image_id, IMAGE_SHOW_WINDOW_X, IMAGE_SHOW_WINDOW_Y)
    cv2.imshow(dtype_image_id, image)

    # Show cut images
    move_x = IMAGE_SHOW_WINDOW_X + image.shape[1]
    move_y = IMAGE_SHOW_WINDOW_Y 
    preproc_x = IMAGE_SHOW_WINDOW_X + 500
    preproc_y = IMAGE_SHOW_WINDOW_Y  + 500
    for i, (sub_img_id, sub_image) in enumerate(sub_images.items()):
        logging.info('Showing sub-image: {}'.format(sub_img_id))
        cv2.namedWindow(sub_img_id)
        # Calculate the window move locaiton.
        # keep adding the width of the sub-image to the move_x co-ord
        move_x += sub_image.shape[1] + 125
        print(move_x)
        cv2.moveWindow(sub_img_id, move_x, move_y)
        cv2.imshow(sub_img_id, sub_image)

        # If set, show the pre-processed images as well
        if SHOW_PREPROCESS:
            preproc_name = 'preproc {}'.format(sub_img_id) 
            preproc_img = aap.preprocess(sub_image)
            cv2.namedWindow(preproc_name)
            preproc_x += 256
            cv2.moveWindow(preproc_name, preproc_x, preproc_y)
            cv2.imshow(preproc_name, aap.preprocess(sub_image))



def show_images_detailed(csv_info, images_dict):
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
        image_full_fn = images_dict[image_id]['full_path']
        image = cv2.imread(image_full_fn)
        # Get annoted image and all of the sub-images
        image, sub_images = get_main_and_sub_images(image, bounding_boxes)
        # Now show the main and sub images
        show_main_and_sub_images(dtype_image_id, image, sub_images)
        print('Press \'n\' for next image. Image will auto forward in 10 sec')
        if cv2.waitKey(25000) == ord('n'):
            continue
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()

def main():
    ''' Main program '''
    # init Stuff
    args = parse_args()
    csv_info = oi_utils.challenge_csv_info(load_from_pickle = True)

    # Now process
    images_list = [x for x in args['images'].split(',')]
    images_list = [x.strip() for x in images_list]
    images_full_fn = get_images_full_path(images_list)
    if not images_full_fn:
        logging.error('Specified images do not exist in any of the tran/va/test'
                      ' directories. Noting to do. Exiting')
    else:
        show_images_detailed(csv_info, images_full_fn)

if __name__ == '__main__':
    main()
