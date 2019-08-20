''' Processes iWild 2019 train images to train and validation files then converts them to tfe records
    Also converts test images to tfe records '''

import pandas as pd
import shutil

from attrdict import AttrDict
from pathlib import Path

# Local imports
from pylibs.ml.preprocessing.train_val_utils import split_train_val
from pylibs.ml.io import images2tferecord
from pylibs.io.file_utils import dir2filelist


# Constants
HOME_DIR = Path('/data/data1/datasets/kaggle/iWildCam_2019/input')
TRAIN_CSV_FN = HOME_DIR/'train.csv'
TEST_CSV_FN = HOME_DIR/'test.csv'
TRAIN_IMAGES_DIR = HOME_DIR/'train_images'
TRAIN_SPLIT_TRAIN_DIR = HOME_DIR/'train_split_train_images'
TRAIN_SPLIT_VAL_DIR = HOME_DIR/'train_split_val_images'
TEST_IMAGES_DIR = HOME_DIR/'test_images'
TFE_RECORDS_DIR = HOME_DIR/'TFERecords'

# Define the Id to animal mapping
ID_TO_ANIMAL = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel',
                4: 'rodent', 5: 'small_mammal', 6: 'elk',
                7: 'pronghorn_antelope', 8: 'rabbit', 9: 'bighorn_sheep',
                10: 'fox', 11: 'coyote', 12: 'black_bear', 13: 'raccoon',
                14: 'skunk', 15: 'wolf', 16: 'bobcat', 17: 'cat',
                18: 'dog', 19: 'opossum', 20: 'bison',
                21: 'mountain_goat', 22: 'mountain_lion', }


def split_train_to_train_val():
    ''' From the train_images dir, split the files to train and val
        sets and copy them to the appropriate dirs '''

    print(f'Getting files list from {TRAIN_IMAGES_DIR}')
    flist = dir2filelist(TRAIN_IMAGES_DIR)


def read_train_csv():
    print('Dim')


def main():
    # First read in the train and test csv's
    train_full_csv = pd.read_csv(TRAIN_CSV_FN)
    test_csv = pd.read_csv(TEST_CSV_FN)

    # Split the full traiun set to X (inputs) and y (labels)
    train_full_y = train_full_csv.category_id
    train_full_X = train_full_csv.drop('category_id', axis=1)

    # now further split it to train and valis sets
    train_X, val_X, train_y, val_y = split_train_val(train_full_X,
                                                     train_full_y)

    # In the _X dataframes, create another column that has the full filename
    train_X['full_file_name'] = train_X.file_name.apply(lambda x: TRAIN_IMAGES_DIR/x)
    val_X['full_file_name'] = val_X.file_name.apply(lambda x: TRAIN_IMAGES_DIR/x)

    # In the _y (labels) dataframes, create another column with the
    # animal names
    train_y = pd.DataFrame(train_y)
    val_y = pd.DataFrame(val_y)
    train_y['category_name'] = train_y.category_id.apply(lambda x: ID_TO_ANIMAL[x])
    val_y['category_name'] = val_y.category_id.apply(lambda x: ID_TO_ANIMAL[x])

    '''
    # Now lets try processing the train images to TFE records
    name = 'train'
    filenames = train_X.full_file_name.to_list()
    filenames = [str(x) for x in filenames]
    texts = train_y.category_name.to_list()
    labels = train_y.category_id.to_list()
    num_shards = 100
    num_threads = 2
    output_directory = TFE_RECORDS_DIR
    images2tferecord.process_image_files(name, filenames, texts, labels,
                                         num_shards, num_threads,
                                         output_directory)

    # Now lets try processing the val images to TFE records
    name = 'val'
    filenames = val_X.full_file_name.to_list()
    filenames = [str(x) for x in filenames]
    texts = val_y.category_name.to_list()
    labels = val_y.category_id.to_list()
    num_shards = 100
    num_threads = 2
    output_directory = TFE_RECORDS_DIR
    images2tferecord.process_image_files(name, filenames, texts, labels,
                                         num_shards, num_threads,
                                         output_directory)
    '''

    # Now lets try processing the train images to TFE records
    name = 'test'
    test_csv['full_file_name'] = test_csv.file_name.apply(lambda x: TEST_IMAGES_DIR/x) 
    filenames = test_csv.full_file_name.to_list()
    filenames = [str(x) for x in filenames]
    texts = ['Unknown'] * len(filenames)
    labels =  [0] * len(filenames)
    num_shards = 100
    num_threads = 2
    output_directory = TFE_RECORDS_DIR
    images2tferecord.process_image_files(name, filenames, texts, labels,
                                         num_shards, num_threads,
                                         output_directory)

if __name__ == '__main__':
    main()
