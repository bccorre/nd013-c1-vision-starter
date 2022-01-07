import argparse
from glob import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger, get_dataset


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    # first clean up existing files
    data_dir_list = data_dir.split("/")
    output_dir = "/".join(data_dir_list[:-3]) if data_dir_list[-1] == "" else "/".join(data_dir_list[:-2])
    train_path = os.path.join(output_dir,"train")
    val_path = os.path.join(output_dir,"test")
    test_path = os.path.join(output_dir,"val")
    if os.path.exists(train_path): # remove folder and contents if exists
        shutil.rmtree(train_path)
    if os.path.exists(val_path):
        shutil.rmtree(val_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
        
    # create new directories
    os.mkdir(train_path)
    os.mkdir(val_path)
    os.mkdir(test_path)
    
    #get list of records to use
    record_files = glob(data_dir+"*.tfrecord")
    
    # read list of selected files, previously selected in the exploratory data analysis
    sel_files = open("selected_files.txt","r")
    
    # get the list of files not in the selected list
    record_list = []
    for r_file in record_files:
        if r_file not in sel_files:
            record_list.append(r_file)
    n_records = len(record_list)
    
    # shuffle the record list
    random.shuffle(record_list)
    n_train = 0 # count how many records are used for training
    n_val = 0
    n_test = 0
    
    # split the files into train, test, and validation
    # for training it is used 75% of the records
    # for validation 15%
    # and for test 10%
    for idx, element in enumerate(record_list):
        filename = element.split("/")[-1]
        if idx // np.ceil(0.75 * n_records) == 0:
            # add to training folder
            os.symlink(element, train_path + "/" + filename)
            n_train += 1
        elif idx // np.ceil(0.90 * n_records) == 0:
            # add to validation folder
            os.symlink(element, val_path + "/" + filename)
            n_val += 1
        else:
            # add to test folder
            os.symlink(element, test_path + "/" + filename)
            n_test += 1
            
    # split the seledted files into all sets keeping the same proportion
    sel_files_sz = len(list(sel_files))
    for idx, element in enumerate(sel_files):
        filename = element.split("/")[-1]
        if idx // np.ceil(0.75 * sel_files_sz) == 0:
            # add to training folder
            os.symlink(element, train_path + "/" + filename)
            n_train += 1
        elif idx // np.ceil(0.90 * sel_files_sz) == 0:
            # add to validation folder
            os.symlink(element, val_path + "/" + filename)
            n_val += 1
        else:
            # add to test folder
            os.symlink(element, test_path + "/" + filename)
            n_test += 1
    
    print(n_train,n_val,n_test)
        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)