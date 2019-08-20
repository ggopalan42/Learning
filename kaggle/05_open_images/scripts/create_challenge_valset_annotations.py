#! /usr/bin/env python

VAL_FILE = 'challenge-2018-image-ids-valset-od.csv'
# VAL_FILE = 'valset_100.csv'
TRAIN_ANNOTATIONS_FILE = 'challenge-2018-train-annotations-bbox_expanded_fixed.csv'
VAL_OUT_FILE = 'challenge-2018-valset-annotations-bbox_expanded_fixed.csv'
HEADER = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'


if __name__ == '__main__':
    write_fh = open(VAL_OUT_FILE, 'wt')
    # Write out the header
    write_fh.write(HEADER)
     
    print('Reading in validation file')
    with open(VAL_FILE) as vfh:
        val_lines = vfh.readlines()
        val_lines = [x.strip() for x in val_lines]
        val_length = len(val_lines)

    print('Reading in train annotations file')
    with open(TRAIN_ANNOTATIONS_FILE) as tfh:
        train_anno_lines = tfh.readlines()
        train_anno_lines = [x.strip() for x in train_anno_lines]

    for i, image_id in enumerate(val_lines[1:]):
        percent_done = (float(i)/float(val_length)) *100
        print('Processing image: {}. {} of {}. {}% done'
                               .format(image_id, i, val_length, percent_done))
        for anno_line in train_anno_lines:
            if anno_line.startswith(image_id):
                write_fh.write('{}\n'.format(anno_line))

    write_fh.close()
