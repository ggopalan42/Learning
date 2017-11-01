#!/usr/bin/env python

# coding: utf-8

# import libraries
import os
import sys

import tsa_cnn_utils as cu

#------------------------------------------------------------------------------
# Constants
# INPUT_FOLDER:                 The folder that contains the source data
# PREPROCESSED_DATA_FOLDER:     The folder that contains preprocessed .npy files
# STAGE1_LABELS:                The CSV file containing the labels by subject
# THREAT_ZONE:                  Threat Zone to train on 
#                                                  (actual number not 0 based)
# BATCH_SIZE:                   Number of Subjects per batch
# EXAMPLES_PER_SUBJECT          Number of examples generated per subject
# FILE_LIST:                    A list of the preprocessed .npy files to batch
# TRAIN_TEST_SPLIT_RATIO:       Ratio to split the FILE_LIST between train 
#                                                                     and test
# TRAIN_SET_FILE_LIST:          The list of .npy files to be used for training
# TEST_SET_FILE_LIST:           The list of .npy files to be used for testing
# IMAGE_DIM:                    The height and width of the images in pixels
# LEARNING_RATE                 Learning rate for the neural network
# N_TRAIN_STEPS                 The number of train steps (epochs) to run
# TRAIN_PATH                    Place to store the tensorboard logs
# MODEL_PATH                    Path where model files are stored
# MODEL_NAME                    Name of the model files
#------------------------------------------------------------------------------
HOME_DIR = os.path.expanduser('~')
TSA_BASE_DIR = os.path.join(HOME_DIR, 'work/datasets/kaggle/TSA')
TSA_DATA_BASE_DIR = os.path.join(TSA_BASE_DIR, 'stage1/inputs')
BODY_ZONES = os.path.join(TSA_DATA_BASE_DIR, 'body_zones.png')
STAGE1_LABELS = os.path.join(TSA_DATA_BASE_DIR, 'stage1_labels.csv')
INPUT_FOLDER = os.path.join(TSA_DATA_BASE_DIR, 'aps')
THREAT_ZONE = 1
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
OUTPUT_PATH = os.path.join(TSA_BASE_DIR, 'stage1/outputs/')
TRAIN_PATH = os.path.join(TSA_BASE_DIR, 'stage1/outputs/tsa_logs/train/')
MODEL_PATH = os.path.join(TSA_BASE_DIR, 'stage1/outputs/tsa_logs/model/')
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE,
                                          IMAGE_DIM, IMAGE_DIM, THREAT_ZONE ))

# Configure GPU Parameters
# Set to run on GPU1 only:
os.environ["CUDA_VISIBLE_DEVICES"]="1" # (or "0" or "" for no-GPU)
# Uncomment below if you want to configure memory on GPU
# config = tf.ConfigProto( gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), device_count = {'GPU': 2})
# set_session(tf.Session(config=config))

# Constants
# Preprocess data using one of three data input choices: 
#      'all_labelled':  All Labelled Data
#      'all':           All data - labelled and unlabelled
#      'expt1':         Predefined set of data - for experiments

DATA_OPTION = 'expt1'

# TSA CNN Class. This class pretty much contains all of the constants at this 
# point
class TsaCNN():
    def __init__(self):
        # Base dirs
        self.home_dir = os.path.expanduser('~')
        self.tsa_base_dir = os.path.join(self.home_dir, 
                                                 'work/datasets/kaggle/TSA')
        self.tsa_data_base_dir = os.path.join(self.tsa_base_dir, 
                                                            'stage1/inputs')
        # Inputs
        self.body_zones_png = os.path.join(self.tsa_data_base_dir, 
                                                           'body_zones.png')
        self.labelled_csv = os.path.join(self.tsa_data_base_dir, 
                                                          'stage1_labels.csv')
        self.input_images_aps = os.path.join(self.tsa_data_base_dir, 'aps')

        # Outputs
        self.output_dir = os.path.join(self.tsa_base_dir, 'stage1/outputs/')
        self.train_dir = os.path.join(self.output_dir, 'tsa_logs/train/')
        self.model_dir = os.path.join(self.output_dir, '/tsa_logs/model/')
        self.model_name = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', 
                           self.learning_rate, self.image_dim, self.image_dim, 
                           self.threat_zone ))
        # Preprocessed
        self.pre_train_dir = os.path.join(self.tsa_base_dir, 
                                         'stage1/preprocessed/gg_cnn_expt1')
        self.pre_pred_dir = os.path.join(self.tsa_base_dir, 
                             'stage1/preprocessed/gg_cnn_expt1/predictions')
        self.data_option = 'expt1'

        # Parameters - including hyper params
        self.threat_zone = 1
        self.batch_size = 16
        self.examples_per_subject = 182

        self.image_dim = 250
        self.learning_rate = 1e-3
        self.n_train_steps = 1

if __name__ == '__main__':

    print('Finished all imports. Starting preprocess')

    '''
    # preprocessed_data_folder = os.path.join(TSA_BASE_DIR,
    #                       'stage1/preprocessed/gg_cnn_expt1',data_option)

    # preprocess_tsa_data(preprocessed_data_folder, data_option)
    # get_train_test_file_list(preprocessed_data_folder, data_option)
    # model_file_name = '{}-tz-{}.{}'.format(DATA_OPTION, THREAT_ZONE, 'tflearn')
    # model_save_file = os.path.join(OUTPUT_PATH, model_file_name)
    # train_conv_net(preprocessed_data_folder, data_option, model_save_file)

    subjects_for_predictions = cu.load_subjects_for_prediction()
    for s in subjects_for_predictions:
        print(s)
    '''
