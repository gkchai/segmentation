## After images and masks are placed in the train and test folders, this script generates the csv
## index files for the train and test folders. Separate val folder is generated from a split of
## train image files

# usage: python create_csv.py

import glob
import os
import csv
import numpy as np
np.random.seed(314) # change this to create new split

# Further split training into training and validation dataset
CREATE_VALIDATION = True
RATIO_VALIDATION = 0.1


DATASET_PATH = "dataset"
PATHS = ["dataset/train", "dataset/test"]
SETS = [["isic17_128", "isic18_128", "usr_mob_128", "usr_d415_128", "tones_128"],
        ["isic17_128", "usr_mob_128", "tones_128"]]
# SETS = [["isic17_256", "isic18_256", "usr_mob_256", "usr_d415_256"], ["isic17_256", "usr_mob_256"]]

if not os.path.exists(DATASET_PATH + "/val"):
    os.mkdir(os.path.join(DATASET_PATH, 'val'))


for idx, mode in enumerate(['train', 'test']):

    for set in SETS[idx]:

        IMG_DIR = os.path.join(PATHS[idx], "image", set)
        MASK_DIR = os.path.join(PATHS[idx], "mask", set)

        def gen_img_list():
            for filename in glob.glob(os.path.join(IMG_DIR, "*.png")):
                yield os.path.basename(filename)

        with open(("dataset/{}/{}.csv").format(mode, set), 'w') as f:

            writer = csv.writer(f)
            if mode == 'train' and CREATE_VALIDATION:
                f_val = open(("dataset/{}/{}.csv").format('val', set), 'w')
                writer_val = csv.writer(f_val)

            cnt_train, cnt_val, cnt = 0, 0, 0
            for img_f in gen_img_list():

                row = [os.path.join(IMG_DIR, img_f), os.path.join(MASK_DIR, img_f)]
                if mode == 'train' and CREATE_VALIDATION:

                    # main logic for splitting into train and val
                    if np.random.rand() > RATIO_VALIDATION:
                        writer.writerow(row)
                        cnt_train += 1
                    else:
                        writer_val.writerow(row)
                        cnt_val += 1
                else:
                    writer.writerow(row)
                    cnt += 1

        if mode == 'train' and CREATE_VALIDATION:
            print("{}: Total = {} ==> Train = {}, Val = {}".format(set, cnt_train+cnt_val, cnt_train, cnt_val))
        else:
            print("{}: Test = {}".format(set, cnt))
