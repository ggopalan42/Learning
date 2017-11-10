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
'''
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
'''

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

DATA_OPTION = 'all_labelled'

# TSA CNN Class. This class pretty much contains all of the constants at this 
# point
class TsaCNN():
    ''' Constants for TSA Challenge CNN simulations '''
    def __init__(self):
        # Some inits
        cu.random_set_seed()

        # Parameters - including hyper params
        self.DATA_OPTION = DATA_OPTION
        self.BATCH_SIZE = 16
        self.EXAMPLES_PER_SUBJECT = 182
        self.TRAIN_TEST_SPLIT_RATIO = 0.2

        self.IMAGE_DIM = 250
        self.LEARNING_RATE = 1e-3
        self.N_TRAIN_STEPS = 10

        # Base dirs
        self.HOME_DIR = os.path.expanduser('~')
        self.TSA_BASE_DIR = os.path.join(self.HOME_DIR, 
                                                 'work/datasets/kaggle/TSA')
        self.TSA_DATA_BASE_DIR = os.path.join(self.TSA_BASE_DIR, 
                                                            'stage1/inputs')
        # Inputs
        self.BODY_ZONES_PNG = os.path.join(self.TSA_DATA_BASE_DIR, 
                                                           'body_zones.png')
        self.LABELLED_CSV = os.path.join(self.TSA_DATA_BASE_DIR, 
                                                          'stage1_labels.csv')
        self.INPUT_IMAGES_DIR = os.path.join(self.TSA_DATA_BASE_DIR, 'aps')

        # Outputs
        self.OUTPUT_DIR = os.path.join(self.TSA_BASE_DIR, 'stage1/outputs/')
        self.TRAIN_DIR = os.path.join(self.OUTPUT_DIR, 'tsa_logs/train/')
        self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, 'tsa_logs/model/')

        # Preprocessed
        self.PRE_TRAIN_DIR = os.path.join(self.TSA_BASE_DIR, 
                                         'stage1/preprocessed/gg_cnn_expt1')
        # PRE_DATA_DIR is where the actual preprocessed data is stored
        self.PRE_DATA_DIR = os.path.join(self.PRE_TRAIN_DIR, self.DATA_OPTION)
        self.PRE_PRED_DIR = os.path.join(self.TSA_BASE_DIR, 
                             'stage1/preprocessed/gg_cnn_expt1/predictions')

class TsaParam():
    ''' This class contains various malleable parameters used in the training 
        and prediction process '''
    def __init__(self):
        # Some inits
        self.file_list = []
        self.train_set_file_list = []
        self.test_set_file_list = []

        # The threat zone to train on
        self.threat_zone = 5

        # CNN Model used
        self.cnn_model = 'alexnet-v0.1'

    # some methods
    def get_model_name(self, tc):
        self.model_name = 'tsa-{}-lr-{}-{}-{}-tz-{}'.format(self.cnn_model, 
                           tc.LEARNING_RATE, tc.IMAGE_DIM, tc.IMAGE_DIM, 
                           self.threat_zone )
        return self.model_name


if __name__ == '__main__':

    print('Finished all imports. Starting preprocess')
    tsa_constants = TsaCNN()
    tsa_params = TsaParam()

    #### Preprocess the training data - needs to done only once ###
    '''
    # train_subject_list = cu.get_train_subject_list(tsa_constants)
    cu.preprocess_tsa_data(train_subject_list, tsa_constants)
    '''

    #### Train on the data and store the trained models ####
    # Get the train/test (validate?) split
    cu.get_train_test_file_list(tsa_constants, tsa_params)
    # Now train on that data
    cu.train_conv_net(tsa_constants, tsa_params)
    print('Done Training!!!!!')

    ##### Predict on the test data #####
    pred_subject_list = cu.get_pred_subject_list(tsa_constants)
    # pred_subject_list = ['0043db5e8c819bffc15261b1f1ac5e42']
    # Preprocess the subjects for predictions
    cu.preprocess_tsa_data(pred_subject_list, tsa_constants, pred_flag=True)

    # Now load the preprocessed prediction data and start predictions
    subject_prediction = cu.predict_tsa_subjects(pred_subject_list, 
                                                  tsa_constants, tsa_params)
    '''
    for p in subject_prediction['0043db5e8c819bffc15261b1f1ac5e42']:
        print(p)
    '''


    # Summarize the prediction for each of the subject
    sub_pred_means = cu.sumarize_tsa_subjects(subject_prediction)
    for s in sub_pred_means:
        print('Subject: {}. Means: {}'.format(s, sub_pred_means[s]))
