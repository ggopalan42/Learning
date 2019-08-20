#! /usr/bin/env python

import os
import sys
import shutil
import random
import logging

from collections import defaultdict

# Challenge datasets constants
USER_HOME_DIR = '/home2/ggopalan/'
DATASETS_BASE_DIR = os.path.join(USER_HOME_DIR,'datasets/kaggle/open_images/')
CLASS_DESCRIPTIONS_FILE = os.path.join(DATASETS_BASE_DIR, 
                         'csv_files/challenge-2018-class-descriptions-500.csv')
BBOX_HIERARCHY_IN_FILE = os.path.join(DATASETS_BASE_DIR, 
                                              'bbox_labels_500_hierarchy.json')
BBOX_HIERARCHY_OUT_FILE = os.path.join(DATASETS_BASE_DIR, 
                                           'bbox_labels_500_hierarchy_hr.json')

# Set logging level
logging.basicConfig(level=logging.INFO)

def convert_csv_to_dict():
    ''' Convert the class descriptions to dict '''
    class_desc_dict = {}
    with open(CLASS_DESCRIPTIONS_FILE) as fh:
        for line in fh:
            class_desc_dict[line.split(',')[0]] = line.split(',')[1].strip()
    return class_desc_dict

def convert_bbox_labels_to_hr(class_desc_dict):
    ''' Convert the bbox_labels hierarchy to human readable '''
    with open(BBOX_HIERARCHY_OUT_FILE, 'wt') as write_fh:
        with open(BBOX_HIERARCHY_IN_FILE) as read_fh:
            for in_line in read_fh:
                # Need to keep in_line as is so the spacing may be preserved
                in_line_strip = in_line.strip()
                if 'LabelName' in in_line_strip:
                    # If this is the line that neewd to be replaced
                    # Get the key
                    label_name_key = in_line_strip.split(':')[1]
                    # Strip out all the junk before and after so it can be used
                    label_name_key = label_name_key.strip()
                    label_name_key = label_name_key.strip('"')
                    label_name_key = label_name_key.strip('",')
                    # Sometimes the key does not exist in class descriptions
                    try:
                        replacement_str = class_desc_dict[label_name_key]
                    except KeyError:
                        replacement_str = label_name_key
                        logging.info('Class key: {} does not exist'
                                                       .format(label_name_key))
                    write_fh.write(in_line.replace(label_name_key,
                                                             replacement_str))
                else:
                    write_fh.write(in_line)

def main():
    ''' Main program '''
    class_desc_dict = convert_csv_to_dict()
    convert_bbox_labels_to_hr(class_desc_dict)

if __name__ == '__main__':
    main()
