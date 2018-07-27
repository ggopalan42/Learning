#! /usr/bin/env python

import os
import sys
import shutil
import random
import logging

from collections import defaultdict

# Challenge datasets constants
USER_HOME_DIR = '/home2/ggopalan/'
DATASETS_BASE_DIR = os.path.join(USER_HOME_DIR,'datasets/kaggle/open_images/' )
# Note: In this case, this train dir contains *only* train images. 
# The reccomended validation set has been moved into the validation dir
# That is it contains: 1643042 images (1743042 - 100000)
CHALLENGE2018_TRAIN_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_train')
CHALLENGE2018_VAL_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_val')
CHALLENGE2018_TEST_DIR = os.path.join(DATASETS_BASE_DIR, 'challenge2018_test')

# Tiny datasets constants
TINY_DATASETS_BASE_DIR=os.path.join(USER_HOME_DIR, 
                                'datasets/kaggle/open_images/tiny_datasets')
NEW_TINY_DATASET_DIR = os.path.join(TINY_DATASETS_BASE_DIR, 'oid_tiny1')
TINY_TRAIN_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'train')
TINY_VAL_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'val')
TINY_TEST_DIR = os.path.join(NEW_TINY_DATASET_DIR, 'test')
# The below constants define the number of images in the train/val/test set
TINY_TRAIN_COUNT = 1000
TINY_VAL_COUNT = 100
TINY_TEST_COUNT = 100
# Below defines the algo to choose the tiny dataset files
TINY_DATASET_CHOOSE_ALGO = 'random'

# csvs
CSVS_BASE_DIR = os.path.join(USER_HOME_DIR, 
                                   'datasets/kaggle/open_images/csv_files')
CSV_BBOX_ANNO_FILE = os.path.join(CSVS_BASE_DIR, 
                                 'challenge-2018-train-annotations-bbox.csv')
CSV_BBOX_IMAGELABELS_FILE = os.path.join(CSVS_BASE_DIR, 
                    'challenge-2018-train-annotations-human-imagelabels.csv')
CSV_BBOX_ANNO_HEADER = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside'
CSV_BBOX_IMAGELABELS_HEADER = 'ImageID,Source,LabelName,Confidence'

# Other constants
RANDOM_SEED = 42

# Set logging level
logging.basicConfig(level=logging.INFO)

# Set random seed
random.seed(RANDOM_SEED)

def choose_random(files_list, files_count):
    ''' Choose files_count files at random from files_list and return them '''
    return random.sample(files_list, files_count)

# The dict below maps choose_algo names to the actual functions
choose_algo_to_fn_map = {
    'random': choose_random,
}

def choose_files(files_list, files_count, choose_algo):
    ''' Select subset of files from files_list based on choose algo '''
    ret_val = choose_algo_to_fn_map[choose_algo](files_list, files_count)
    return ret_val

def make_needed_dirs():
    ''' Make the neeeded dirs for the new tiny datasets '''
    logging.info('making train/val/test dirs if they do not exist')
    for tiny_dir in [TINY_TRAIN_DIR, TINY_VAL_DIR, TINY_TEST_DIR]:
        if not os.path.exists(tiny_dir):
            os.makedirs(tiny_dir)

def make_tiny_train_set(source_dir, dest_dir, image_count, choose_algo):
    ''' Choose image_count number of files from the source_dir, select
        them according to the choose_algo and copy them to the dest dir '''
    logging.info('    Selecting and copying over {} images to tiny train dir'
                                                       .format(image_count))
    source_files = os.listdir(source_dir)
    copy_files = choose_files(source_files, image_count, choose_algo)

    # Now copy them over
    for image_file in copy_files:
        src_file = os.path.join(source_dir, image_file)
        shutil.copy(src_file, dest_dir)
    logging.info('    Copying of tiny train subset is complete')

def make_tiny_val_set(source_dir, dest_dir, image_count, choose_algo):
    ''' Choose image_count number of files from the source_dir, select
        them according to the choose_algo and copy them to the dest dir '''
    logging.info('    Selecting and copying over {} images to tiny val dir'
                                                       .format(image_count))
    source_files = os.listdir(source_dir)
    copy_files = choose_files(source_files, image_count, choose_algo)

    # Now copy them over
    for image_file in copy_files:
        src_file = os.path.join(source_dir, image_file)
        shutil.copy(src_file, dest_dir)
    logging.info('    Copying of tiny val subset is complete')

def make_tiny_test_set(source_dir, dest_dir, image_count, choose_algo):
    ''' Choose image_count number of files from the source_dir, select
        them according to the choose_algo and copy them to the dest dir '''
    logging.info('    Selecting and copying over {} images to tiny test dir'
                                                       .format(image_count))
    source_files = os.listdir(source_dir)
    copy_files = choose_files(source_files, image_count, choose_algo)

    # Now copy them over
    for image_file in copy_files:
        src_file = os.path.join(source_dir, image_file)
        shutil.copy(src_file, dest_dir)
    logging.info('    Copying of tiny test subset is complete')

def make_tiny_datasets():
    ''' Make tiny train/val/test datasets based on the specified algorithm '''
    logging.info('Starting copying of files to tiny dataset')
    make_tiny_train_set(CHALLENGE2018_TRAIN_DIR, TINY_TRAIN_DIR, 
                                    TINY_TRAIN_COUNT, TINY_DATASET_CHOOSE_ALGO)
    make_tiny_val_set(CHALLENGE2018_VAL_DIR, TINY_VAL_DIR, TINY_VAL_COUNT,
                                                    TINY_DATASET_CHOOSE_ALGO)
    make_tiny_test_set(CHALLENGE2018_TEST_DIR, TINY_TEST_DIR, TINY_TEST_COUNT,
                                                    TINY_DATASET_CHOOSE_ALGO)
    logging.info('Done copying of files to tiny dataset')

def make_tiny_dtype_csvs(bbox_dict, imagelabels_dict, src_dir, dst_dir, dtype):
    ''' Create the subset of csv files for tiny train/val dataset '''
    # get the image ids from the source dir
    src_files_list = os.listdir(src_dir)
    src_image_ids = [x.split('.')[0] for x in src_files_list]

    # Now create the tiny <dtype> bbox csv file
    # Open file for writing
    write_fn = os.path.join(dst_dir, 
               'challenge2018-tiny-{}-annotations-bbox.csv'.format(dtype))
    logging.info( '    Writing tiny {} bbox subset to: {}'
                                                 .format(dtype, write_fn))

    with open(write_fn, 'wt') as write_fh:
        # Write the header
        write_fh.write(CSV_BBOX_ANNO_HEADER)
        for image_id in src_image_ids:
            for line in bbox_dict[image_id]:
                write_fh.write(line)

    # Now create the tiny <dtype> human-imagelabels csv file
    # Open file for writing
    write_fn = os.path.join(dst_dir, 
      'challenge2018-tiny-{}-annotations-human-imagelabels.csv'.format(dtype))
    logging.info( '    Writing tiny {} human-imagelables subset to: {}'
                                                 .format(dtype, write_fn))

    with open(write_fn, 'wt') as write_fh:
        # Write the header
        write_fh.write(CSV_BBOX_IMAGELABELS_HEADER)
        for image_id in src_image_ids:
            for line in imagelabels_dict[image_id]:
                write_fh.write(line)

def convert_csv_to_dict():
    ''' Convert the bbox and imagelabels csvs to dicts. Otherwise processing
        them takes too long '''
    # Using default dicts. Uninited dict key will be an empty list
    bbox_dict = defaultdict(lambda: [])
    imagelabels_dict = defaultdict(lambda: [])

    # Create the bbox_dict
    logging.info('Converting the entire bbox csv into a dict')
    with open(CSV_BBOX_ANNO_FILE) as fh:
        for line in fh:
            if line.startswith('ImageID'):
                continue                  # Skip the header
            else:
                image_id = line.split(',')[0]
                bbox_dict[image_id].append(line)

    # Create the humanlabels dict
    logging.info('Converting the entire human-imagelabels csv into a dict')
    with open(CSV_BBOX_IMAGELABELS_FILE) as fh:
        for line in fh:
            if line.startswith('ImageID'):
                continue                  # Skip the header
            else:
                image_id = line.split(',')[0]
                imagelabels_dict[image_id].append(line)

    return bbox_dict, imagelabels_dict

def make_tiny_csvs(bbox_dict, imagelabels_dict):
    ''' Make corresponding csv files for the dataset '''
    # Make the train CSV subset
    logging.info('Making tiny train csvs')
    make_tiny_dtype_csvs(bbox_dict, imagelabels_dict, 
                                TINY_TRAIN_DIR, NEW_TINY_DATASET_DIR, 'train')
    # Make the val CSV subset
    logging.info('Making tiny val csvs')
    make_tiny_dtype_csvs(bbox_dict, imagelabels_dict, 
                                TINY_VAL_DIR, NEW_TINY_DATASET_DIR, 'val')

def main():
    ''' Main program '''
    make_needed_dirs()
    make_tiny_datasets()
    bbox_dict, imagelabels_dict = convert_csv_to_dict()
    make_tiny_csvs(bbox_dict, imagelabels_dict)

if __name__ == '__main__':
    main()
