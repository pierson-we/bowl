import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import skimage.io
import urllib.request
import shutil
import cv2
import pandas as pd
from skimage import exposure
from skimage import transform
import matplotlib.pyplot as plt
from scipy import ndimage

############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = []
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes.append({'class': 1, 'y1': y1, 'x1': x1, 'y2': y2, 'x2': x2})
    return boxes

def make_borders(img):
    img = np.where(img >= 0.5, 0, 1) #this is the mask for the water/non water raster
    
    #Here I get the Land and NoData edges
    # struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(img) #, struct)
    edges = img ^ erode
    return np.where(edges >= 0.5, 1, 0) 

def format_mask(mask, img, _id, num_classes):
    mask_multiclass = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.bool)
    bg = np.reshape(np.where(mask==0, 1, 0), (mask.shape[0], mask.shape[1]))
    if np.mean(img) < 150:
        img_class = 0
    else:
        img_class = 1
    # img_class = int(class_df.loc[_id]['classes'])
    mask_multiclass[:,:,img_class] = np.reshape(mask, (mask.shape[0], mask.shape[1]))
    mask_multiclass[:,:,int(img_class+(num_classes/2))] = bg
    return mask_multiclass

def make_json(train_path, img_size): # , test_path, img_size, classes_csv):
    # class_df = pd.read_csv(classes_csv, header=0, index_col='_id')
    # num_classes = len(class_df.classes.unique().tolist()) * 2
    num_classes = 1
    train_ids = next(os.walk(train_path))[1]
    # test_ids = next(os.walk(test_path))[1]
    training = [] # = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    # Y_train = np.zeros((len(train_ids), img_size, img_size, num_classes), dtype=np.bool)
    for i, id_ in enumerate(train_ids):
        train_dict = {}
        path = train_path + id_
        train_dict['filename'] = path + '/images/' + id_ + '.png'
        train_dict['shape'] = [img_size, img_size, 4]
        # img = cv2.imread(path + '/images/' + id_ + '.png')
        # img = cv2.resize(img, (img_size, img_size))
        # X_train[i] = img
        # mask = np.zeros((img_size, img_size, num_classes), dtype=np.bool)
        mask = []
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            mask_ = cv2.resize(mask_, (img_size, img_size))
            mask_ = mask_[:, :, np.newaxis]
            # mask = np.maximum(mask, format_mask(mask_, img, id_, num_classes))
            mask.append(mask_)
        mask = np.reshape(np.array(mask), (img_size, img_size, -1))
        train_dict['boxes'] = []
        for box in extract_bboxes(mask):
            train_dict['boxes'].append(box)
        training.append(train_dict)
        # Y_train[i] =  make_2_class(mask) # make_3_class(mask)
        #p_low, p_high = np.percentile(img, (1, 99))
        #img = exposure.rescale_intensity(img, in_range=(p_low, p_high))
        #X_train[i] = img
        
        # if np.sum(img) < 7.5e07:
        #     X_train[i] = img * (10./np.mean(img))
        # elif np.sum(img) < 1.6e08:
        #     X_train[i] = img * (83./np.mean(img))
        # else:
        #     X_train[i] = img * (105./np.mean(img))

        # print(Y_train[i].shape)
    # X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
	#     sizes_test = []
	#     X_test = pd.DataFrame(index=test_ids)
	#     X_test['ImageID'] = test_ids
	#     X_test['images'] = None
	#     X_test['images_raw'] = None
	#     X_test['shape'] = None

	#     for i, id_ in enumerate(test_ids):
	#         path = test_path + id_
	#         img = cv2.imread(path + '/images/' + id_ + '.png')
	#         X_test.loc[id_]['shape'] = (img.shape[0], img.shape[1])
	#         # sizes_test.append([img.shape[0], img.shape[1]])
	#         X_test.loc[id_]['images_raw'] = img
	#         img = cv2.resize(img, (img_size, img_size))
	#         p_low, p_high = np.percentile(img, (1, 99))
	#         img = exposure.rescale_intensity(img, in_range=(p_low, p_high))
	#         # X_test[i] = img
	#         # if np.sum(img) < 7.5e07:
	#         #     X_test.loc[id_]['images'] = img * (10./np.mean(img))
	#         # elif np.sum(img) < 1.6e08:
	#         #     X_test.loc[id_]['images'] = img * (83./np.mean(img))
	#         # else:
	#         #     X_test.loc[id_]['images'] = img * (105./np.mean(img))

	#         X_test.loc[id_]['images'] = img
    return training

def get_train_weights(xtr):
    train_weights = np.zeros(xtr.shape[0])
    for i in range(0, xtr.shape[0]):
        img = xtr[i]
        if np.mean(img) < 70:
            train_weights[i] = 1
        elif np.mean(img) < 190:
            train_weights[i] = 1
        else:
            train_weights[i] = 1
    return train_weights

# Define generator. Using keras ImageDataGenerator. You can change the method of data augmentation by changing data_gen_args.
from keras.preprocessing.image import ImageDataGenerator


def generator(xtr, xval, ytr, yval, batch_size):
    # data_gen_args = dict(horizontal_flip=True,
    #                      vertical_flip=True,
    #                      rotation_range=90.,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.1)
    # image_datagen = ImageDataGenerator(**data_gen_args)
    # mask_datagen = ImageDataGenerator(**data_gen_args)
    # image_datagen.fit(xtr, seed=7)
    # mask_datagen.fit(ytr, seed=7)
    # image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
    # mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)
    # train_generator = zip(image_generator, mask_generator)

    # val_gen_args = dict()
    # image_datagen_val = ImageDataGenerator(**val_gen_args)
    # mask_datagen_val = ImageDataGenerator(**val_gen_args)
    # image_datagen_val.fit(xval, seed=7)
    # mask_datagen_val.fit(yval, seed=7)
    # image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
    # mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
    # val_generator = zip(image_generator_val, mask_generator_val)

    def simple_gen(x, y, batch_size):
        while True:
            count = 0
            for sample in range(0, x.shape[0]):
                count+=1
                x_sample, y_sample = np.expand_dims(x[sample,:,:,:], 0), np.expand_dims(y[sample,:,:,:], 0)
                if ((count-1) % batch_size == 0) or count == 1:
                    x_batch, y_batch = x_sample, y_sample
                else:
                    x_batch, y_batch = np.concatenate((x_batch, x_sample)), np.concatenate((y_batch, y_sample))
                if count % batch_size == 0:
                    yield x_batch, y_batch

    train_generator = simple_gen(xtr, ytr, batch_size)
    val_generator = simple_gen(xval, yval, batch_size)
    return train_generator, val_generator

from skimage.morphology import label # label regions
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)

def resize_masks(row):
    # mask = np.average(row['masks'][:,:,0:2], axis=-1)
    row['masks_resized'] = np.where(transform.resize(image=row['masks'], output_shape=(row['shape'][0],row['shape'][1]), preserve_range=True, mode='constant') >= 0.5, 1, 0)
    return row

from skimage.morphology import closing, opening, disk
def clean_img(x):
    # return opening(closing(x, disk(1)), disk(3))
    return x

# from crf import crf
# def format_preds(Y_train, test_img_df):
#     # test_img_df = crf(Y_train, test_img_df)
#     test_img_df = test_img_df.apply(resize_masks, axis=1)
#     test_img_df['rles'] = test_img_df['masks_resized'].map(clean_img).map(lambda x: list(prob_to_rles(x)))
#     print(test_img_df.head())
#     global out_pred_list
#     out_pred_list = []
#     test_img_df.columns
#     def create_out_pred_list(row):
#         global out_pred_list
#         for c_rle in row['rles']:
#             out_pred_list+=[{'ImageId': row['ImageID'], 
#                                  'EncodedPixels': ' '.join(np.array(c_rle).astype(str))}]
#     test_img_df.apply(create_out_pred_list, axis=1)
#     out_pred_df = pd.DataFrame(out_pred_list)
#     print(out_pred_df.shape[0], 'regions found for', test_img_df.shape[0], 'images')
#     # out_pred_df.sample(3)
#     # n_img = 3
#     # for i in range(0,10):   
#     #     fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))
#     #     for (_, d_row), (c_im, c_lab, c_clean) in zip(test_img_df.sample(n_img).iterrows(), m_axs):
#     #         c_im.imshow(d_row['images'])
#     #         c_im.axis('off')
#     #         c_im.set_title('Microscope')
            
#     #         c_lab.imshow(d_row['masks'])
#     #         c_lab.axis('off')
#     #         c_lab.set_title('Predicted Raw')
            
#     #         c_clean.imshow(clean_img(d_row['masks_resized']))
#     #         c_clean.axis('off')
#     #         c_clean.set_title('Clean')
#     #     plt.show()

#     return test_img_df, out_pred_df

def masks_1_class(mask_multiclass, test_img):
    mask = mask_multiclass[:,:,img_class]
    num_classes = mask_multiclass.shape[-1]
    mask = mask_multiclass[:,:,0]
    mask = np.argmax(mask_multiclass, axis=2)
    mask = np.reshape(mask, (mask_multiclass.shape[0], mask_multiclass.shape[1])).astype(np.uint8)
    
    # for i in range(1, num_classes):
    #     mask = np.maximum(mask, mask_multiclass[:,:,i])
    
    # max_pixels = 0
    # for i in range(0, num_classes):
    #     summed_pixels = np.sum(mask_multiclass[:,:,i])
    #     if summed_pixels > max_pixels:
    #         max_pixels = summed_pixels
    #         test_class = i
    return mask

def decluster_masks(df):
    df['masks'] = None
    def process_mask(row):
        num_classes = row.masks_multiclass.shape[-1]/2

        if np.mean(row.images) < 150:
            img_class = 0
        else:
            img_class = 1

        row.masks = row.masks_multiclass[:,:,img_class]
        return row
    return df.apply(process_mask, axis=1)

def lr_schedule(epoch):
    initial = 0.0001
    step_reduction = 0.1
    step_number = epoch / 10
    lr = initial * (step_reduction**step_number)
    return lr
