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
import pickle

import tensorflow as tf
import tsahelper as tsa

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

from timeit import default_timer as timer

def random_set_seed():
    ''' Set the seed for repeatable results from different runs '''
    random.seed(42)

# def get_train_subject_list(labelled_data_file, input_folder, data_option):
def get_train_subject_list(tc):

    ''' Depending on the data option, return a list of subjects '''

    print('Getting subject list for data option: {}'.format(tc.DATA_OPTION))
    if tc.DATA_OPTION == 'all_labelled':
        # OPTION 1: get a list of all subjects for which there are labels
        df = pd.read_csv(tc.LABELLED_CSV)
        df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
        subject_list = df['Subject'].unique().tolist()
    elif tc.DATA_OPTION == 'all':
        # OPTION 2: get a list of all subjects for whom there is data
        subject_list = [os.path.splitext(subject)[0] for subject in 
                                               os.listdir(tc.INPUT_IMAGES_DIR)]
    elif tc.DATA_OPTION == 'expt1':
        # OPTION 3: get a list of subjects for small bore test purposes
        subject_list = ['00360f79fd6e02781457eda48f85da90',
                        '0043db5e8c819bffc15261b1f1ac5e42',
                        '0050492f92e22eed3474ae3a6fc907fa',
                        '006ec59fa59dd80a64c85347eef810c7',
                        '0097503ee9fa0606559c56458b281a08',
                        '011516ab0eca7cad7f5257672ddde70e']
    elif tc.DATA_OPTION.startswith('random'):
        # OPTION 4: get a list of random subjects from the labelled subjects
        # First read in the labelled subjects file
        df = pd.read_csv(tc.LABELLED_CSV)
        df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
        subject_list = df['Subject'].unique().tolist()
        # Choose a random set of subjects. Count specified by the random_##, the ## part
        random_count = int(tc.DATA_OPTION.split('_')[1])
        return random.sample(subject_list, random_count)
    else:
        print('ERROR: Data option: {} not supported. Returning'
                                                        .format(data_option))
        return False

    return subject_list

def get_pred_subject_list(tc):
    ''' This returns the subject lists for which predictions need to be made '''

    # Get list of labelled images
    df = pd.read_csv(tc.LABELLED_CSV)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    labelled_list = df['Subject'].unique()

    all_list = [os.path.splitext(subject)[0] for subject in 
                                               os.listdir(tc.INPUT_IMAGES_DIR)]
    return list(set(all_list) - set(labelled_list))


def read_processes_image(subject_name, tc, tp):
    ''' Read a single subject, processes it through the image pipe and return
        a numpy array of the image '''

    image = tsa.read_data(os.path.join(tc.INPUT_IMAGES_DIR,
                                                       subject_name+'.aps'))
    # transpose so that the slice is the first dimension shape(16, 620, 512)
    image = image.transpose()

    # Process image for that particular threat zone
    tz_num = tp.threat_zone
    threat_zone = tsa.zone_slice_list[tz_num]
    crop_dims = tsa.zone_crop_list[tz_num]
    threat_zone_images = []

    for img_num, img in enumerate(image):
        print('Processing threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                
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
            
            # append the features and labels to this threat 
            # zone's example array
            print ('-> appending final image to list')
            threat_zone_images.append(zero_centered_img)

    return threat_zone_images



# ## The Preprocessor
#--------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets
# parameters:      subject_list: List to preprocess
#                  tc: TSA Constants
#                  pred_flag: If true, preprocess for prediction
# returns:         none
#--------------------------------------------------------------------------
def preprocess_tsa_data(subject_list, tc, pred_flag=False):
    

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
        images = tsa.read_data(os.path.join(tc.INPUT_IMAGES_DIR,subject+'.aps'))

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone 
        # and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(
                                 zip(tsa.zone_slice_list, tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            if pred_flag:
                label = np.array([0,0])
            else:
                label = np.array(tsa.get_subject_zone_label(tz_num, 
                             tsa.get_subject_labels(tc.LABELLED_CSV, subject)))

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

                    # append the features and labels to this threat 
                    # zone's example array
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
                              (tc.BATCH_SIZE * tc.EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                if pred_flag:
                    pre_out_dir = tc.PRE_PRED_DIR
                else:
                    pre_out_dir = tc.PRE_DATA_DIR

                out_fn = os.path.join(pre_out_dir,
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

            if pred_flag:
                pre_out_dir = tc.PRE_PRED_DIR
            else:
                pre_out_dir = tc.PRE_DATA_DIR

            out_fn = os.path.join(pre_out_dir,
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

# ## Train and Test Split
# ----------------------------------------------------------------------------
# get_train_test_file_list(): gets the batch file list, 
#                             splits between train and test
# parameters:      
#                  tc: TSA Constants
#                  tp: TSA Parameters (not a very nice name. Need to think of something else)
# returns:         none
#-----------------------------------------------------------------------------

def get_train_test_file_list(tc, tp):

    print('Looking for pre-processed data in: {}'
                                         .format(tc.PRE_DATA_DIR))

    if os.listdir(tc.PRE_DATA_DIR) == []:
        print ('No preprocessed data available.'
                                        'Skipping preprocessed data setup..')
    else:
        tp.file_list = [f for f in os.listdir(tc.PRE_DATA_DIR) 
               if re.search(re.compile('-tz' + str(tp.threat_zone) + '-'), f)]

        # print('File list: {}'.format(FILE_LIST))
        train_test_split = len(tp.file_list) -  \
                     max(int(len(tp.file_list) * tc.TRAIN_TEST_SPLIT_RATIO),1)
        tp.train_set_file_list = tp.file_list[:train_test_split]
        tp.test_set_file_list = tp.file_list[train_test_split:]
        print('Train/Test split: {}'.format(train_test_split))
        print('Train file list: {}'.format(tp.train_set_file_list))
        print('Test file list: {}'.format(tp.test_set_file_list))
        print('Train/Test Split -> {} file(s) of {} used for testing'.format( 
              len(tp.file_list) - train_test_split, len(tp.file_list)))
        
# ## Get File List
# ----------------------------------------------------------------------------
# get_tz_file_list(): gets the batch file list for specified threat zone 
# parameters:      
#                  tc: TSA Constants
#                  tp: TSA Parameters (not a very nice name. Need to think of something else)
#                  tz: Threat Zone
#                  pred_flag: If True, load files from prediction folder
# returns:         List of files for specified threat zone
#-----------------------------------------------------------------------------

def get_tz_file_list(tc, tp, tz, pred_flag=True):

    if pred_flag:
        file_list_dir = tc.PRE_PRED_DIR
    else:
        file_list_dir = tc.PRE_DATA_DIR

    print('Looking for pre-processed data in: {}'.format(file_list_dir))

    if os.listdir(file_list_dir) == []:
        print ('No preprocessed data available. Returning Null')
        return None
    else:
        file_list = [f for f in os.listdir(file_list_dir) 
               if re.search(re.compile('-tz' + str(tz) + '-'), f)]
    return file_list

def get_checkpoint_model_name(tc):
    ''' Get the latest model name from the checkpoint file '''
    checkpoint_file = os.path.join(tc.MODEL_DIR, 'checkpoint')
    with open(checkpoint_file, 'rt') as fh:
        checkpoint_dict = dict(x.rstrip().split(':') for x in fh)
    # At this point, returning the model name keyed by 'model_checkpoint_path'
    # Figure out later if this is the correct key
    return checkpoint_dict['model_checkpoint_path'].strip().replace('"','')

def get_tz_model_name(tc, tp):
    ''' Get the latest model name from the list of models saved '''
    model_list = os.listdir(tc.MODEL_DIR)
    model_name = tp.get_model_name(tc)
    saved_model_full = next(x for x in model_list if x.startswith(model_name))
    saved_model_name = os.path.splitext(saved_model_full)[0]
    return os.path.join(tc.MODEL_DIR, saved_model_name)

        
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
  
# ## Shuffling the Training Set
#----------------------------------------------------------------------------
# shuffle_train_set(): shuffle the list of batch files so that each train step
#                      receives them in a different order since the TRAIN_SET_FILE_LIST
#                      is a global
# parameters:      train_set - the file listing to be shuffled
# returns:         none
#----------------------------------------------------------------------------
def shuffle_train_set(train_set, tp):
    sorted_file_list = random.shuffle(train_set)
    # tp.train_set_file_list = sorted_file_list
    
# ## Defining the Alexnet CNN
#-----------------------------------------------------------------------------
# alexnet(width, height, lr): defines the alexnet
#
# parameters:      width - width of the input image
#                  height - height of the input image
#                  lr - learning rate
#                  tc - TSA constants
#
# returns:         none
#
#-----------------------------------------------------------------------------

def alexnet(width, height, lr, tc, tp):
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

    model = tflearn.DNN(network, tensorboard_dir=tc.TRAIN_DIR, 
                  tensorboard_verbose=3, max_checkpoints=1,
             checkpoint_path=os.path.join(tc.MODEL_DIR, tp.get_model_name(tc)))
    return model


# ## The Trainer
#-----------------------------------------------------------------------------
# train_conv_net(): runs the train op
# parameters:   tc: TSA Constants
#               tp: TSA Parameters (not a very nice name. Need to think of something else)

# returns:      none. Trained model is saved
#-----------------------------------------------------------------------------
def train_conv_net(tc, tp):
    
    val_features = []
    val_labels = []
    
    # get train and test batches
    get_train_test_file_list(tc, tp)
    
    # instantiate model
    model = alexnet(tc.IMAGE_DIM, tc.IMAGE_DIM, tc.LEARNING_RATE, tc, tp)
    
    # read in the validation test set
    for j, test_f_in in enumerate(tp.test_set_file_list):
        if j == 0:
            val_features, val_labels = input_pipeline(test_f_in, 
                                                           tc.PRE_DATA_DIR)
        else:
            tmp_feature_batch, tmp_label_batch = input_pipeline(test_f_in, 
                                                            tc.PRE_DATA_DIR)
            val_features = np.concatenate((tmp_feature_batch, val_features), 
                                                                     axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)
    val_features = val_features.reshape(-1, tc.IMAGE_DIM, tc.IMAGE_DIM, 1)
    
    # start training process
    for i in range(tc.N_TRAIN_STEPS):

        # shuffle the train set files before each step
        shuffle_train_set(tp.train_set_file_list, tp)
        print(tp.train_set_file_list)
        
        # run through every batch in the training set
        for f_in in tp.train_set_file_list:
            
            # read in a batch of features and labels for training
            feature_batch, label_batch = input_pipeline(f_in, tc.PRE_DATA_DIR)
            feature_batch = feature_batch.reshape(-1, tc.IMAGE_DIM, 
                                                            tc.IMAGE_DIM, 1)
            print ('Feature Batch Shape ->', feature_batch.shape)                
            # run the fit operation
            print('Test set size: {}'.format(len(tp.test_set_file_list)))
            print('Train set size: {}'.format(len(tp.train_set_file_list)))
            print('Validation size: {}'.format(len(val_features)))
            print('Train size: {}'.format(len(feature_batch)))

            model.fit({'features': feature_batch}, {'labels': label_batch}, 
                      n_epoch=1, 
                      validation_set=({'features': val_features}, 
                                      {'labels': val_labels}), 
                      shuffle=True, snapshot_step=None, show_metric=True, 
                      run_id=tp.get_model_name(tc))
    '''
    if output_model:
        print('Saving trained model to: {}'.format(output_model))
        model.save(output_model)
    '''

# ## The Predictor
#-----------------------------------------------------------------------------
# predict_tsa_subjects(): predicts from the trained model
# parameters:   subject_list: List of subjects to predict on
#               tc: TSA Constants
#               tp: TSA Parameters (not a very nice name. Need to think of something else)

# returns:      none. Trained model is saved
#-----------------------------------------------------------------------------
def predict_tsa_subjects(subject_list, tc, tp):
    
    val_features = []
    val_labels = []
    
    # instantiate model
    model = alexnet(tc.IMAGE_DIM, tc.IMAGE_DIM, tc.LEARNING_RATE, tc, tp)

    # Load the model for the current tz
    # model_name = get_checkpoint_model_name(tc)
    model_name = get_tz_model_name(tc, tp)
    print('Loading model: {}'.format(model_name))
    model.load(model_name)

    subject_prediction_dict={}
    for subject in subject_list:
        # Get the processed image for this threat zone
        subject_processed_img = read_processes_image(subject, tc, tp)

        print('Predicting for subject: {}'.format(subject))
        subject_prediction_dict[subject]=[]

        # Now go through each image slice in the pre-processed image for 
        # the subject and predict based on the threat zone loaded model
        for img in subject_processed_img:
            prediction = model.predict(img.reshape(-1, tc.IMAGE_DIM, 
                                                             tc.IMAGE_DIM, 1))
            subject_prediction_dict[subject].append(prediction)

    '''
    for s in subject_prediction_dict:
        for p in subject_prediction_dict[s]:
            print('Pred for sub: {} is: {}'.format(s, p))
    '''
    return subject_prediction_dict

def sumarize_tsa_subjects(subject_prediction_dict):
    ''' Get an average of the predictions for each subject. 
        At a later date, a more enlightned algo can be used '''
    subject_prediction_summary = {}
    for subject in subject_prediction_dict:
        subject_mean = np.concatenate(subject_prediction_dict[subject]).mean(axis=0)
        subject_prediction_summary[subject] = subject_mean
    '''
    for s in subject_prediction_summary:
        print('Subject: {}. Means: {}'.format(s, subject_prediction_summary[s]))
    '''
    return subject_prediction_summary

