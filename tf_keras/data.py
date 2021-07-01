import os
import pdb
import glob
import cv2
from keras.backend import learning_phase
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def adjust_data(img, label, data, cnt, val ='F'):
    ''''' Adjust images and labels using data flag for network inputs
    Args:
        img   (np.array):   Augmented images
        label (np.array):   Augmented labels
        data  (str):        Data Flag
        cnt   (int):        Data Identification
        val'''
    
    img = img / 255
    if data == 'complete':
        label[label < 25] = 0
        label[label >= 25] = 1
        label = np.concatenate(((-label) + 1, label), axis = -1)
    
    elif data == 'core':
        l1 = (label < 25).astype(np.uint8)
        l2 = (label < 75).astype(np.uint8)
        label1 = np.logical_and(l1, l2).astype(np.uint8)
        l1 = (label > 125).astype(np.uint8)
        l2 = (label < 175).astype(np.uint8)
        label2 = np.logical_and(l1, l2).astype(np.uint8)
        label = np.logical_or(label1, label2).astype(np.uint8)
        label = np.concatenate(((-label)+ 1, label), axis = -1)
    
    elif data == 'enhancing':
        l1 = (label > 125).astype(np.uint8)
        l2 = (label < 175).astype(np.uint8)
        label = np.logical_and(l1, l2).astype(np.uint8)
        label = np.concatenate(((-label)+ 1), axis = -1)

    else:
        raise ValueError('data flag error!')

    return img, label

def datasets(args, mode = 'train', image_color_mode = "grayscale", label_color_mode = "grayscale", image_save_prefix = "image", label_save_prefix = "label", save_to_dir = None, target_size = (240,240), seed = 1):
    ''' Prepare dataset ( pre-processing + augmentation(optional) )
    Args:
        args (argparse):          Arguments parsered in command-lind
        mode (str):               Mode ('train', 'valid', 'test')
        image_color_mode (str):   Image color Mode Flag
        label_color_mode (str):   Label color Mode Flag
        image_save_prefix (str):  Prefix to use for filnames of saved images
        label_save_prefix (str):  Prefix to use for filename of saved labels
        save_to_dir (str):        Save directory
        target_size (tuple):      Target Size
        seed (int):               Seed value
    '''
    if mode == 'train':
        shuffle = True
        image_datagen = ImageDataGenerator(rotation_range = 20, horizontal_flip = True, vertical_flip = True, width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.2, zoom_range = 0.1)
        label_datagen = ImageDataGenerator(rotation_range = 20, horizontal_flip = True, vertical_flip = True, width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.2, zoom_range = 0.1)
    
    elif mode == 'test' or mode == 'valid':
        shuffle = False
        image_datagen = ImageDataGenerator()
        label_datagen = ImageDataGenerator()

    else:
        raise ValueError('datasets mode error!')
    
    image_generator1 = image_datagen.flow_from_directory(
        args.image_root,
        classes = [args.image_folder1],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = args.batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        shuffle = shuffle,
        seed = seed
    )
    label_generator1 = label_datagen.flow_from_directory(
        args.label_root,
        classes = [args.label_folder1],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = args.batch_size,
        save_to_dir = save_to_dir,
        save_prefix = label_save_prefix,
        shuffle = shuffle,
        seed = seed
    )
    data_generator1 = zip(image_generator1, label_generator1)
    cnt = 0