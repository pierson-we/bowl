import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
import sys
from skimage.io import imread
from skimage import transform
import skimage
import matplotlib.pyplot as plt
import seaborn as sns
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

print('loading data...')
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

def read_and_stack(in_img_list):
    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
 
test_df = img_df.query('TrainingSplit=="test"')
test_rows = []
group_cols = ['Stage', 'ImageId']
for n_group, n_rows in test_df.groupby(group_cols):
    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()
    test_rows += [c_row]
test_img_df = pd.DataFrame(test_rows)    
test_img_df['images'] = test_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:3])

def rescale(img):
    p1, p99 = np.percentile(img, (0.1, 99))
    img = skimage.exposure.rescale_intensity(img, in_range=(p1, p99))
    return img
# test_img_df['images'] = test_img_df['images'].map(rescale)

# imlist = test_img_df['images'].tolist()
# features = zeros([len(imlist), 512])
# for img in imlist:
    

n_img = 4
row = 0
for i in range(0,(test_img_df.shape[0]/(n_img*3) + 1)):   
    fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))
    for c_im1, c_im2, c_im3, c_im4 in m_axs:
        c_im1.imshow(test_img_df.loc[row]['images'])
        c_im1.axis('off')
        # c_im1.set_title('microscope')
        row += 1
        if row == test_img_df.shape[0]: break

        c_im2.imshow(test_img_df.loc[row]['images'])
        c_im2.axis('off')
        # c_im2.set_title('microscope')
        row += 1
        if row == test_img_df.shape[0]: break

        c_im3.imshow(test_img_df.loc[row]['images'])
        c_im3.axis('off')
        # c_im3.set_title('microscope')
        row += 1
        if row == test_img_df.shape[0]: break

        c_im4.imshow(test_img_df.loc[row]['images'])
        c_im4.axis('off')
        # c_im4.set_title('microscope')
        row += 1
        if row == test_img_df.shape[0]: break
    plt.show()