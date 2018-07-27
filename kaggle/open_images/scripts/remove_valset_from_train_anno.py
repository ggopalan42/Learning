#! /usr/bin/env python

# VAL_FILE = 'challenge-2018-image-ids-valset-od.csv'
VAL_FILE = 'valset_100.csv'
TRAIN_ANNOTATIONS_FILE = 'challenge-2018-train-annotations-bbox_expanded_fixed.csv'
TRAIN_ANNOTATIONS_OUT_FILE = 'challenge-2018-train-annotations-wo-valset-bbox_expanded_fixed.csv'
HEADER = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'


if __name__ == '__main__':
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
