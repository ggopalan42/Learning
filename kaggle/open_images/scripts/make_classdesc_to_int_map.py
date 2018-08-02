#! /usr/bin/env python

''' This is a rather simple program that converts each class description
    to an int. I realize this could easily have been done in bash.
    The main purpose of this is to store the labels of images as a numpy
    int32 in hdf5. hdf5 does not handle strings very well '''

import os
import sys
import logging

# Challenge datasets constants
USER_HOME_DIR = '/home2/ggopalan/'
CSVS_BASE_DIR = os.path.join(USER_HOME_DIR, 
                                   'datasets/kaggle/open_images/csv_files')
CHALLENGE_CLASS_DESC_FILE = os.path.join(CSVS_BASE_DIR, 
                               'challenge-2018-class-descriptions-500.csv')
CHALLENGE_CLASS_DESC_INT_MAP_FILE = os.path.join(CSVS_BASE_DIR, 
                               'challenge-2018-class-desc-to-int-map.csv')

# Set logging level
logging.basicConfig(level=logging.INFO)

def main():
    ''' Main program '''
    start_int = 1000
    logging.info('Start processing class desc csv file')
    with open (CHALLENGE_CLASS_DESC_INT_MAP_FILE, 'wt') as write_fh:
        with open(CHALLENGE_CLASS_DESC_FILE) as read_fh:
            for line in read_fh:
                line_out = '{},{}'.format(line.split(',')[0], start_int)
                start_int += 1
                write_fh.write('{}\n'.format(line_out))
    logging.info('Done processing class desc csv file')

if __name__ == '__main__':
    main()
