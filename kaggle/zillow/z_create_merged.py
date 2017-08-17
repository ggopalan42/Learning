''' 
This .py loads the zillow properties data and the training data, merges them
on parcel id and pickles it.

Date: 8/7/2017

'''

import pandas as pd

def z_merge_n_save():
    ''' This function loads the zillow properties data and the training data, merges them on parcel id and pickles it. '''

    # Load the training set and the entire zillow data set
    train_2016=pd.read_csv('input/train_2016_v2.csv')
    properties_2016=pd.read_csv('properties_2016.csv')

    # For easier understanding of the columns, rename all columns from
    # that in the zillow data
    rename_dict=pd.Series.from_csv('zillow_data_dictionary.csv', 
                                                      header=None).to_dict()
    properties_2016.rename(columns=rename_dict, inplace=True)

    # Merge the dfs on parcelid
    merged_2016=train_2016.merge(properties_2016, how='inner', on='parcelid')

    # Now store it
    merged_2016.to_pickle('merged_2016_v2.pkl')

if __name__ == '__main__':
    z_merge_n_save()
    print('Done!')
