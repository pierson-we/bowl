import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
import sys
from skimage.io import imread
from skimage import transform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage

# import cv2
# %matplotlib inline

dsb_data_dir = os.path.join('/Users/wep/Kaggle/bowl', 'input')
stage_label = 'stage1'

# Read in the labels
# Load the RLE-encoded output for the training set

train_labels = pd.read_csv(os.path.join(dsb_data_dir,'{}_train_labels.csv'.format(stage_label)))
train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda ep: [int(x) for x in ep.split(' ')])
train_labels.sample(3)

# Load in all Images
# Here we load in the images and process the paths so we have the appropriate information for each image
all_images = glob(os.path.join(dsb_data_dir, 'stage1_*', '*', '*', '*'))
img_df = pd.DataFrame({'path': all_images})
img_id = lambda in_path: in_path.split('/')[-3]
img_type = lambda in_path: in_path.split('/')[-2]
img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]
img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]
img_df['ImageId'] = img_df['path'].map(img_id)
img_df['ImageType'] = img_df['path'].map(img_type)
img_df['TrainingSplit'] = img_df['path'].map(img_group)
img_df['Stage'] = img_df['path'].map(img_stage)
img_df.sample(2)


print('loading data...')
train_df = img_df.query('TrainingSplit=="train"')
train_rows = []
group_cols = ['Stage', 'ImageId']
for n_group, n_rows in train_df.groupby(group_cols):
    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
    c_row['masks'] = n_rows.query('ImageType == "masks"')['path'].values.tolist()
    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()
    train_rows += [c_row]
train_img_df = pd.DataFrame(train_rows)
IMG_CHANNELS = 3
def read_and_stack(in_img_list):
    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
def process_input(img):
    bgr = img[:,:,[2,1,0]].astype(np.float32) # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(lab[:,:,0])
    return np.reshape(img, newshape=(img.shape[0], img.shape[1], -1))
def read_and_stack_and_resize(in_img_list):
    return np.where(np.sum(np.stack([transform.resize(image=imread(c_img),output_shape=(256,256), preserve_range=True, mode='constant') for c_img in in_img_list], 0), 0)/255.0 >= 0.5, 1, 0)
def resize_imgs(row):
    row['shape'] = row['images'].shape
    row['images'] = transform.resize(image=row['images'], output_shape=(256,256), preserve_range=True, mode='constant')
    return row
def resize_masks(row):
    row['masks_resized'] = np.where(transform.resize(image=row['masks'], output_shape=(row['shape'][0],row['shape'][1]), preserve_range=True, mode='constant') >= 0.5, 1, 0)
    return row

train_img_df['images'] = train_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])
train_img_df['masks'] = train_img_df['masks'].map(read_and_stack).map(lambda x: x.astype(int))

av_int = train_img_df['images'].map(lambda x: np.ptp(x))
# print(av_int.mean())
# print(av_int.std())
print(av_int)
print(av_int.describe())
# print(av_int.hist(bins=20))
# plt.show

def make_borders(img):
    # B5 = src.read()

    # B4 = B5.reshape(453, 484)
    #I reclass from 3 to 2 classes to get all the borders of the water
    # B3 = np.where(B4 != 1, 0, B4)
    mask = ((img != 1)) #this is the mask for the water/non water raster
    # mask2 = ((B4 != 1)) #this is the mask for the 3 classes (water, land and NoData)

    # #Here I get the NoData edges
    # struct = ndimage.generate_binary_structure(2, 2)
    # erode = ndimage.binary_erosion(B3, struct)
    # edges = mask ^ erode

    #Here I get the Land and NoData edges
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(img, struct)
    edges = mask ^ erode
    return edges

print('calculating borders...')
train_img_df['borders'] = None
train_img_df['borders'] = train_img_df['masks'].map(make_borders)

n_img = 4
row = 0
for i in range(0,(train_img_df.shape[0]/(n_img*3) + 1)):   
    fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))
    for c_im1, c_im2, c_im3, c_im4 in m_axs:
        c_im1.imshow(train_img_df.loc[row]['images'])
        c_im1.axis('off')
        # c_im1.set_title('microscope')

        c_im2.imshow(train_img_df.loc[row]['borders'])
        c_im2.axis('off')
        # c_im2.set_title('microscope')
        row += 1
        if row == train_img_df.shape[0]: break

        c_im3.imshow(train_img_df.loc[row]['images'])
        c_im3.axis('off')
        # c_im3.set_title('microscope')

        c_im4.imshow(train_img_df.loc[row]['borders'])
        c_im4.axis('off')
        # c_im4.set_title('microscope')
        row += 1
        if row == train_img_df.shape[0]: break
    plt.show()