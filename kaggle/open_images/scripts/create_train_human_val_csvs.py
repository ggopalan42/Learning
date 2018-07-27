#! /usr/bin/env python

from collections import defaultdict

VAL_FILE = 'challenge-2018-image-ids-valset-od.csv'
# VAL_FILE = 'valset_100.csv'
TRAIN_HUMAN_FILE = 'challenge-2018-train-annotations-human-imagelabels_expanded_fixed.csv'

VAL_HUMAN_OUT_FILE = 'challenge-2018-valset-annotations-human-imagelabels_expanded_fixed.csv'
TRAIN_HUMAN_OUT_FILE = 'challenge-2018-train-annotations-wo-valset-human-imagelabels_expanded_fixed.csv'
HEADER = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'

def open_all_files():
    val_in_fh = open(VAL_FILE)
    train_human_in_fh = open(TRAIN_HUMAN_FILE)

    val_human_out_fh = open(VAL_HUMAN_OUT_FILE, 'wt')
    train_human_out_fh = open(TRAIN_HUMAN_OUT_FILE, 'wt')

    return val_in_fh, train_human_in_fh, val_human_out_fh, train_human_out_fh

def close_all_files(fh_list):
    for fh in fh_list:
        fh.close
    
def write_headers(val_human_out_fh, train_human_out_fh):
    print('Writing headers to train human out files')
    val_human_out_fh.write(HEADER)
    train_human_out_fh.write(HEADER)

def convert_train_to_dict(train_human_lines):
    print('Converting train annotation to dict')

    train_human_dict = defaultdict(lambda: [])  # Uninited dict key will be an empty list
    for train_line in train_human_lines:
        train_image_id = train_line.split(',')[0]
        # The train line can simply be appended since default dict behaviour is
        # set to return ampty list for new key
        train_human_dict[train_image_id].append(train_line)
    return train_human_dict


def main():
    # Open all files
    print('Opening all files')
    val_in_fh, train_human_in_fh, val_human_out_fh, train_human_out_fh = open_all_files()
    write_headers(val_human_out_fh, train_human_out_fh)

    print('Reading in validation file')
    val_lines = val_in_fh.readlines()
    val_lines = [x.strip() for x in val_lines[1:]]  # [1:] is to exclude header
    val_length = len(val_lines)
    print('Validation file length is {}'.format(val_length))


    print('Reading in train human file')
    train_human_lines = train_human_in_fh.readlines()
    train_human_lines = [x.strip() 
                   for x in train_human_lines[1:]]  # [1:] is to exclude header
    train_human_length = len(train_human_lines)
    print('train human file length is {}'.format(train_human_length))

    train_human_dict = convert_train_to_dict(train_human_lines)

    # Now create the val human set and train human file minus the val set
    for i, image_id in enumerate(val_lines):
        print('Processing image id: {}. {} of {}'.format(image_id, i,val_length))
        if image_id in train_human_dict:
            # Write that out to the validation human file
            print('Writing image_id: {} to human valset'.format(image_id))
            for line in train_human_dict[image_id]:
                val_human_out_fh.write('{}\n'.format(line))

            # Delete that key from the dict
            print('Deleting image_id: {}'.format(image_id))
            del train_human_dict[image_id]

    # Now write out the remainder of the dict 
    print('Now writing train human (w/o valset) ')
    train_human_dict_len = len(train_human_dict)
    for i, key in enumerate(train_human_dict):
        print('Writing image id: {} to train human file. {} or {}'
                                     .format(key, i, train_human_dict_len))
        for line in train_human_dict[key]:
            train_human_out_fh.write('{}\n'.format(line))

    # In the end close all files
    print('Closing all files')
    close_all_files([val_in_fh, train_human_in_fh, val_human_out_fh, train_human_out_fh])

if __name__ == '__main__':

    main()

    ''' Old stuff
    write_fh = open(TRAIN_ANNOTATIONS_OUT_FILE, 'wt')
    # Write out the header
    write_fh.write(HEADER)
     
    print('Reading in validation file')
    with open(VAL_FILE) as vfh:
        val_lines = vfh.readlines()
        val_lines = [x.strip() for x in val_lines[1:]]
        val_length = len(val_lines)

    print('Reading in train annotations file')
    with open(TRAIN_ANNOTATIONS_FILE) as tfh:
        train_anno_lines = tfh.readlines()
        train_anno_lines = [x.strip() for x in train_anno_lines[1:]]
        train_anno_length = len(train_anno_lines)

    for i, anno_line in enumerate(train_anno_lines):
        percent_done = (float(i)/float(train_anno_length)) *100
        anno_image_id = anno_line.split(',')[0]
        print('Processing anno image id: {}. {} of {}. {}% done'
                    .format(anno_image_id, i, train_anno_length, percent_done))
        for image_id in val_lines:
            if anno_line.startswith(image_id):
                del train_anno_lines[i]

    # Write the train minus valset
    write_fh.write("\n".join(train_anno_lines))
    write_fh.close()
    '''
