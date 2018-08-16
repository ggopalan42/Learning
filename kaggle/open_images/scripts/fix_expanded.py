#! /usr/bin/env python

# FILENAME = '../annotations-bbox_exp.csv'
# FILENAME = '../challenge-2018-train-annotations-bbox_expanded.csv'
FILENAME = '../challenge-2018-train-annotations-human-imagelabels_expanded.csv'
# FILENAME = '../human-imagelabels_expanded.csv'

if __name__ == '__main__':

    with open(FILENAME) as fh:
        lines = fh.readlines()
        lines = [x.strip() for x in lines]
        header = lines[0]
        print(header)

        for line in lines:
            if line.startswith('ImageID'):
                pass
            else:
                print(line)
