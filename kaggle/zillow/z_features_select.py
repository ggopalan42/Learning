''' 
This programe reads in the merged zillow dataset. It then reads the
features.yml file and subsets the df with each set of specified features
and writes out the sub-setted df with the name of the feature

Date: 8/11/2017

'''

import os
import yaml
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

yaml_file = 'features_select.yml'
merged_2016_pkl = 'merged_2016_v2.pkl'


def open_merged(pkl_fn):
    return pd.read_pickle(pkl_fn)

def split_df(fn, full_df):
    ''' This function will read in a full (pickled) dataframe and store
        many sub-setted dataframes (in pickle) as specified in the yaml file.'''
    cwd = os.getcwd()
    ffn = os.path.join(cwd, fn)

    # Read in the yam file
    with open(ffn) as ffh:
        fsel = yaml.load(ffh)
    for fname, fdict in fsel.items():
        flist=[w.strip() for w in fdict['features'].split(',')]

        # Start subsetting the df
        sub_df = full_df[flist]

        # Split the df into train and test sets
        split_ratio = fdict['train_test_split']['split_ratio']
        random_state = fdict['train_test_split']['random_state']
        train_df, test_df = train_test_split(sub_df, test_size=split_ratio, random_state=random_state)

        # Now store the subsetted df, with train and test split
        train_fn = os.path.join(cwd, fname+'_train.pkl')
        test_fn = os.path.join(cwd, fname+'_test.pkl')
        print('Saving sub-setted df with name: {} to files {} and {}'.format(fname, train_fn, test_fn))
        print('Size of train set = {} and size of test set = {}'.format(len(train_df), len(test_df)))
        train_df.to_pickle(train_fn)
        test_df.to_pickle(test_fn)


def trans_onehot(df, feature, trans_type):
    print('In one hot with feature: {} and trans_type {}'.format(feature, trans_type['onehot']))


def trans_normalize(df, feature, trans_type):
    print('In normalize with feature: {} and trans_type {}'.format(feature, trans_type['normalize']))


transform_mappings = {
    'onehot': trans_onehot,
    'normalize': trans_normalize
}

def transform_feature(df, feature, trans_type):
    print('Transforming feature: {}'.format(feature))
    tt_key = list(trans_type.keys())[0]    # Not sure if there is a simpler way to do this
    transform_mappings[tt_key](df, feature, trans_type)


def features_transform(yaml_fn, train_test):
    ''' This function will read in many dataframes as specified in the yaml file.
        and transform each feature of the df as specified '''
    # Read in the yaml file
    with open(yaml_fn) as yfh:
        trans_cfg = yaml.load(yfh)

    # Now go through each yaml config and read in the pickled df
    for base_fn, fdict in trans_cfg.items():
        pkl_fn = '{}_{}{}'.format(base_fn, train_test, '.pkl')
        print('Reading DataFrame pickled in: {}'.format(pkl_fn))
        sub_df = pd.read_pickle(pkl_fn)
        print('Size of df in file name {} is {}'.format(pkl_fn, len(sub_df)))
        # Now go through each feature and transform it
        for feature, trans_type in fdict['transform'].items():
            transform_feature(sub_df, feature, trans_type)

if __name__ == '__main__':
    zdf = open_merged(merged_2016_pkl)
    split_df(yaml_file, zdf)

    # Transform the train sets
    features_transform(yaml_file, train_test='train')

