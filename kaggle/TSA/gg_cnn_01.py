#!/usr/bin/env python

# coding: utf-8

# import libraries
import numpy as np 
import pandas as pd
import os
import re
import cv2
import random
import tflearn
import sys

import tensorflow as tf
import tsahelper as tsa

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

from timeit import default_timer as timer

# Configure GPU Parameters
# Set to run on GPU1 only:
os.environ["CUDA_VISIBLE_DEVICES"]="1" # (or "0" or "" for no-GPU)
# Uncomment below if you want to configure memory on GPU
# config = tf.ConfigProto( gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), device_count = {'GPU': 2})
# set_session(tf.Session(config=config))



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
THREAT_LABELS = os.path.join(TSA_DATA_BASE_DIR, 'stage1_labels.csv')
INPUT_FOLDER = os.path.join(TSA_DATA_BASE_DIR, 'aps')
DATASET_OPTION = {'all_labelled': 0, 'all': 0, 'expt1': 1}
STAGE1_LABELS = os.path.join(TSA_DATA_BASE_DIR, 'stage1_labels.csv')
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


# ## The Preprocessor
#--------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets
# parameters:      none
# returns:         none
#--------------------------------------------------------------------------
def preprocess_tsa_data(preprocessed_data_folder, data_option):
    

    print('Preprocessing for data option: {}'.format(data_option))
    if data_option == 'all_labelled':
        # OPTION 1: get a list of all subjects for which there are labels
        df = pd.read_csv(STAGE1_LABELS)
        df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
        subject_list = df['Subject'].unique()
    elif data_option == 'all':
        # OPTION 2: get a list of all subjects for whom there is data
        subject_list = [os.path.splitext(subject)[0] for subject in 
                                                      os.listdir(INPUT_FOLDER)]
    elif data_option == 'expt1':
        # OPTION 3: get a list of subjects for small bore test purposes
        subject_list = ['00360f79fd6e02781457eda48f85da90',
                        '0043db5e8c819bffc15261b1f1ac5e42',
                        '0050492f92e22eed3474ae3a6fc907fa',
                        '006ec59fa59dd80a64c85347eef810c7',
                        '0097503ee9fa0606559c56458b281a08',
                        '011516ab0eca7cad7f5257672ddde70e']
    else:
        print('ERROR: Data option: {} not supported. Returning'
                                                        .format(data_option))
        return False

    # intialize tracking and saving items
    batch_num = 1
    threat_zone_examples = []
    start_time = timer()
    
    for subject in subject_list:
        # read in the images
        print('--------------------------------------------------------------')
        print('t+> {:5.3f} |Reading images for subject #: {}'
                                         .format(timer()-start_time, subject))
        print('--------------------------------------------------------------')
        images = tsa.read_data(os.path.join(INPUT_FOLDER, subject+'.aps'))

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone 
        # and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(
                                 zip(tsa.zone_slice_list, tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num, 
                             tsa.get_subject_labels(STAGE1_LABELS, subject)))

            for img_num, img in enumerate(images):

                print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                print('Threat Zone Label -> {}'.format(label))
                
                if threat_zone[img_num] is not None:

                    # correct the orientation of the image
                    print('-> reorienting base image') 
                    base_img = np.flipud(img)
                    print('-> shape {}|mean={}'.format(base_img.shape, 
                                                       base_img.mean()))

                    # convert to grayscale
                    print('-> converting to grayscale')
                    rescaled_img = tsa.convert_to_grayscale(base_img)
                    print('-> shape {}|mean={}'.format(rescaled_img.shape, 
                                                       rescaled_img.mean()))

                    # spread the spectrum to improve contrast
                    print('-> spreading spectrum')
                    high_contrast_img = tsa.spread_spectrum(rescaled_img)
                    print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                       high_contrast_img.mean()))

                    # get the masked image
                    print('-> masking image')
                    masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                    print('-> shape {}|mean={}'.format(masked_img.shape, 
                                                       masked_img.mean()))

                    # crop the image
                    print('-> cropping image')
                    cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                    print('-> shape {}|mean={}'.format(cropped_img.shape, 
                                                       cropped_img.mean()))

                    # normalize the image
                    print('-> normalizing image')
                    normalized_img = tsa.normalize(cropped_img)
                    print('-> shape {}|mean={}'.format(normalized_img.shape, 
                                                       normalized_img.mean()))

                    # zero center the image
                    print('-> zero centering')
                    zero_centered_img = tsa.zero_center(normalized_img)
                    print('-> shape {}|mean={}'.format(zero_centered_img.shape, 
                                                       zero_centered_img.mean()))

                    # append the features and labels to this threat zone's example array
                    print ('-> appending example to threat zone {}'
                                                               .format(tz_num))
                    threat_zone_examples.append([[tz_num], zero_centered_img, 
                                                                        label])
                    print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(
                                            len(threat_zone_examples),
                                            len(threat_zone_examples[0]),
                                            len(threat_zone_examples[0][0]),
                                            len(threat_zone_examples[0][1][0]),
                                            len(threat_zone_examples[0][1][1]),
                                            len(threat_zone_examples[0][2])))
                else:
                    print('-> No view of tz:{} in img:{}. Skipping to next...'
                                                     .format( tz_num, img_num))
                print('------------------------------------------------')

        # each subject gets EXAMPLES_PER_SUBJECT number of 
        # examples (182 to be exact, so this section just writes out the the 
        # data once there is a full minibatch complete.
        if ((len(threat_zone_examples) % 
                                    (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                out_fn = os.path.join(preprocessed_data_folder,
                         'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'
                         .format(tz_num+1, len(threat_zone_examples[0][1][0]),
                              len(threat_zone_examples[0][1][1]), batch_num))
                print(' -> writing: {}'.format(out_fn))

                # get this tz's examples
                tz_examples = [example for example in threat_zone_examples
                                                    if example[0] == [tz_num]]

                # drop unused columns
                tz_examples_to_save.append([[features_label[1], 
                                             features_label[2]] 
                                            for features_label in tz_examples])

                # save batch.  Note that the trainer looks for tz{} where {} 
                # is a tz_num 1 based in the minibatch file to select which 
                # batches to use for training a given threat zone
                np.save(out_fn, tz_examples_to_save)
                del tz_examples_to_save

            #reset for next batch 
            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1
    
    # we may run out of subjects before we finish a batch, so we write out 
    # the last batch stub
    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(tsa.zone_slice_list):

            tz_examples_to_save = []

            out_fn = os.path.join(preprocessed_data_folder,
                     'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'
                     .format(tz_num+1, len(threat_zone_examples[0][1][0]),
                          len(threat_zone_examples[0][1][1]), batch_num))
            # write out the batch and reset
            print(' -> writing: {}'.format(out_fn))

            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples 
                                                    if example[0] == [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                        for features_label in tz_examples])

            #save batch
            np.save(out_fn, tz_examples_to_save)
# unit test ---------------------------------------
# preprocess_tsa_data()


# ## Train and Test Split
# ----------------------------------------------------------------------------
# get_train_test_file_list(): gets the batch file list, 
#                             splits between train and test
# parameters:      none
# returns:         none
#-----------------------------------------------------------------------------

def get_train_test_file_list(preprocessed_data_folder, data_option):
    global FILE_LIST
    global TRAIN_SET_FILE_LIST
    global TEST_SET_FILE_LIST

    print('Looking for pre-processed data in: {}'
                                         .format(preprocessed_data_folder))

    if os.listdir(preprocessed_data_folder) == []:
        print ('No preprocessed data available. Skipping preprocessed data setup..')
    else:
        FILE_LIST = [f for f in os.listdir(preprocessed_data_folder) 
                    if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]
        # print('File list: {}'.format(FILE_LIST))
        train_test_split = len(FILE_LIST) -                                   \
                              max(int(len(FILE_LIST)*TRAIN_TEST_SPLIT_RATIO),1)
        TRAIN_SET_FILE_LIST = FILE_LIST[:train_test_split]
        TEST_SET_FILE_LIST = FILE_LIST[train_test_split:]
        # print('Train/Test split: {}'.format(train_test_split))
        # print('Train file list: {}'.format(TRAIN_SET_FILE_LIST))
        # print('Test file list: {}'.format(TEST_SET_FILE_LIST))
        print('Train/Test Split -> {} file(s) of {} used for testing'.format( 
              len(FILE_LIST) - train_test_split, len(FILE_LIST)))
        
# ## Generating an Input Pipeline
#-----------------------------------------------------------------------------
# input_pipeline(filename, path): prepares a batch of features and labels 
#                                 for training
# parameters:      filename - the file to be batched into the model
#                  path - the folder where filename resides
# returns:         feature_batch - a batch of features to train or test on
#                  label_batch - a batch of labels related to the feature_batch
#-----------------------------------------------------------------------------
def input_pipeline(filename, path):

    preprocessed_tz_scans = []
    feature_batch = []
    label_batch = []
    
    #Load a batch of preprocessed tz scans
    preprocessed_tz_scans = np.load(os.path.join(path, filename))
        
    #Shuffle to randomize for input into the model
    np.random.shuffle(preprocessed_tz_scans)
    
    # separate features and labels
    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])
            label_batch.append(example[1])
    
    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    label_batch = np.asarray(label_batch, dtype=np.float32)
    
    return feature_batch, label_batch
  
# unit test -----------------------------------------------------------------
'''
print ('Train Set -----------------------------')
for f_in in TRAIN_SET_FILE_LIST:
    feature_batch, label_batch = input_pipeline(f_in, preprocessed_data_folder)
    print (' -> features shape {}:{}:{}'.format(len(feature_batch), 
                                                len(feature_batch[0]), 
                                                len(feature_batch[0][0])))
    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))
    
print ('Test Set -----------------------------')
for f_in in TEST_SET_FILE_LIST:
    feature_batch, label_batch = input_pipeline(f_in, preprocessed_data_folder)
    print (' -> features shape {}:{}:{}'.format(len(feature_batch), 
                                                len(feature_batch[0]), 
                                                len(feature_batch[0][0])))
    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))
'''


# ## Shuffling the Training Set
#----------------------------------------------------------------------------
# shuffle_train_set(): shuffle the list of batch files so that each train step
#                      receives them in a different order since the TRAIN_SET_FILE_LIST
#                      is a global
# parameters:      train_set - the file listing to be shuffled
# returns:         none
#----------------------------------------------------------------------------
def shuffle_train_set(train_set):
    sorted_file_list = random.shuffle(train_set)
    TRAIN_SET_FILE_LIST = sorted_file_list
    
# Unit test ---------------
'''
print ('Before Shuffling ->', TRAIN_SET_FILE_LIST)
shuffle_train_set(TRAIN_SET_FILE_LIST)
print ('After Shuffling ->', TRAIN_SET_FILE_LIST)
'''


# ## Defining the Alexnet CNN
#-----------------------------------------------------------------------------
# alexnet(width, height, lr): defines the alexnet
#
# parameters:      width - width of the input image
#                  height - height of the input image
#                  lr - learning rate
#
# returns:         none
#
#-----------------------------------------------------------------------------

def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum', 
                         loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME, 
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, 
                        max_checkpoints=1)
    return model


# ## The Trainer
#-----------------------------------------------------------------------------
# train_conv_net(): runs the train op
# parameters:      preprocessed_data_folder: dir where preprocessed files are 
#                                            stored
#                  data_option: The type of preprocessed data to use in training
#                  output_folder: If provided, will save the model in this dir
# returns:         none
#-----------------------------------------------------------------------------
def train_conv_net(preprocessed_data_folder, data_option, output_model=None):
    
    val_features = []
    val_labels = []
    
    # get train and test batches
    get_train_test_file_list(preprocessed_data_folder, data_option)
    
    # instantiate model
    model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    
    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_features, val_labels = input_pipeline(test_f_in, 
                                                    preprocessed_data_folder)
        else:
            tmp_feature_batch, tmp_label_batch = input_pipeline(test_f_in, 
                                                    preprocessed_data_folder)
            val_features = np.concatenate((tmp_feature_batch, val_features), 
                                                                       axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)
    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
    
    # start training process
    for i in range(N_TRAIN_STEPS):

        # shuffle the train set files before each step
        shuffle_train_set(TRAIN_SET_FILE_LIST)
        
        # run through every batch in the training set
        for f_in in TRAIN_SET_FILE_LIST:
            
            # read in a batch of features and labels for training
            feature_batch, label_batch = input_pipeline(f_in, 
                                                     preprocessed_data_folder)
            feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
            print ('Feature Batch Shape ->', feature_batch.shape)                
            # run the fit operation
            print('Test set size: {}'.format(len(TEST_SET_FILE_LIST)))
            print('Train set size: {}'.format(len(TRAIN_SET_FILE_LIST)))
            print('Validation size: {}'.format(len(val_features)))
            print('Train size: {}'.format(len(feature_batch)))

            # Run only on GPU1
            model.fit({'features': feature_batch}, {'labels': label_batch}, 
                      n_epoch=1, 
                      validation_set=({'features': val_features}, 
                                      {'labels': val_labels}), 
                      shuffle=True, snapshot_step=None, show_metric=True, 
                      run_id=MODEL_NAME)
    if output_model:
        print('Saving trained model to: {}'.format(output_model))
        model.save(output_model)
            
# unit test -----------------------------------
# train_conv_net()

def load_subjects_for_prediction():
    ''' This returns the subject lists for which predictions need to be made '''

    # Get list of labelled images
    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    labelled_list = df['Subject'].unique()

    all_list = [os.path.splitext(subject)[0] 
                                     for subject in os.listdir(INPUT_FOLDER)]
    return list(set(all_list) - set(labelled_list))

if __name__ == '__main__':

    print('Finished all imports. Starting preprocess')
    # Preprocess data using one of three data input choices: 
    #      'all_labelled':  All Labelled Data
    #      'all':           All data - labelled and unlabelled
    #      'expt1':         Predefined set of data - for experiments

    data_option = 'all_labelled'
    preprocessed_data_folder = os.path.join(TSA_BASE_DIR,
                            'stage1/preprocessed/gg_cnn_expt1',data_option)


    # preprocess_tsa_data(preprocessed_data_folder, data_option)
    # get_train_test_file_list(preprocessed_data_folder, data_option)
    model_file_name = '{}-tz-{}.{}'.format(data_option, THREAT_ZONE, 'tflearn')
    model_save_file = os.path.join(OUTPUT_PATH, model_file_name)
    # train_conv_net(preprocessed_data_folder, data_option, model_save_file)

    subjects_for_predictions = load_test_set()
    # predict_threats(
