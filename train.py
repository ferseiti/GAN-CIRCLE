import tensorflow as tf
import argparse
import numpy as np
import os
import sys
import h5py
# import pydicom
import scipy.misc

from models.cyclegan import CycleGAN
from util.parser import training_parser

os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():
    args = training_parser().parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)
    name = args.name
    restore = args.restore
    restore_ckpt = True if restore else False

    # f = h5py.File('/media/extern-drive/Mayo_data/Mayo_train3_2D.h5', 'r')pip ppasd
    #f = h5py.File('/media/external-drive/Data/Mayo/2_times_noise_downsample_20db_same_size/Mayo_train_2nds_2D_new.h5', 'r')
    f = h5py.File('/ddn/beamline/Fernando/upscaling/talitas/0008.h5')
    data = f.get('data')  # input size 64*64
    label = f.get('label')  # label size 64*64

    args.w = data.shape[1]
    args.h = data.shape[2]
    args.c = data.shape[3] #channels

    args.ow = label.shape[1]
    args.oh = label.shape[2]
    args.oc = label.shape[3] #channels
    print(data.shape)
    #File paths
    train_dir = os.path.join('Network/', name)

    cyclegan = CycleGAN(args, True, restore_ckpt)
    cyclegan.train(data.value, label.value)
    f.close()
    
if __name__ == '__main__':
    main()

