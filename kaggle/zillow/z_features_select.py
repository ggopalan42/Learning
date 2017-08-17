''' 
This programe reads in the merged zillow dataset. It then reads the
features.yml file and subsets the df with each set of specified features
and writes out the sub-setted df with the name of the feature

Date: 8/11/2017

'''

import pandas as pd
import yaml
import os

yaml_file = 'features_select.yml'
merged_2016_pkl = 'merged_2016_v2.pkl'


def open_merged(pkl_fn):
    return pd.read_pickle(pkl_fn)

def process_yaml(fn, full_df):
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

        # Now store the subsetted df
        sub_fn = os.path.join(cwd, fname+'.pkl')
        print('Saving sub-setted df with name: {} to file {}'.format(fname, sub_fn))
        sub_df.to_pickle(sub_fn)


def trans_onehot(df, feature):
    print('In one hot with feature: {}'.format(feature))


def trans_normalize(df, feature):
    print('In normalize with feature: {}'.format(feature))


transform_mappings = {
    'onehot': trans_onehot,
    'normalize': trans_normalize
}

def transform_feature(df, feature, trans_type):
    print('Transforming feature: {} to transform type: {}'.format(feature, trans_type))
    transform_mappings[trans_type](df, feature)


def features_transform(yaml_fn):
    ''' This function will read in many dataframes as specified in the yaml file.
        and transform each feature of the df as specified '''
    # Read in the yaml file
    with open(yaml_fn) as yfh:
        trans_cfg = yaml.load(yfh)

    # Now go through each yaml config and read in the pickled df
    for fname, fdict in trans_cfg.items():
        print('Reading DataFrame pickled in: {}'.format(fname+'.pkl'))
        sub_df = pd.read_pickle(fname+'.pkl')
        # Now go through each feature and transform it
        for feature, trans_type in fdict['transform'].items():
            transform_feature(sub_df, feature, trans_type)

if __name__ == '__main__':
    zdf = open_merged(merged_2016_pkl)
    # process_yaml(yaml_file, zdf)
    features_transform(yaml_file)

