import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
import sys
from skimage.io import imread
from skimage import transform
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
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

# Create Training Data
# Here we make training data and load all the images into the dataframe. We take a simplification here of grouping all the regions together (rather than keeping them distinct).
print('\ncreating training data...')
# %%time
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
    # return np.sum(np.stack([cv2.resize(imread(c_img),dsize=(128,128)) for c_img in in_img_list], 0), 0)/255.0
    # img = np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
def process_input(img):
    bgr = img[:,:,[2,1,0]].astype(np.float32) # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(lab[:,:,0])
    return np.reshape(img, newshape=(img.shape[0], img.shape[1], -1))
def read_and_stack_and_resize(in_img_list):
    return np.where(np.sum(np.stack([transform.resize(image=imread(c_img),output_shape=(256,256), preserve_range=True, mode='constant') for c_img in in_img_list], 0), 0)/255.0 >= 0.5, 1, 0)
    # img = np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
    # return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0
def resize_imgs(row):
	row['shape'] = row['images'].shape
	row['images'] = transform.resize(image=row['images'], output_shape=(256,256), preserve_range=True, mode='constant')
	return row
def resize_masks(row):
	row['masks_resized'] = np.where(transform.resize(image=row['masks'], output_shape=(row['shape'][0],row['shape'][1]), preserve_range=True, mode='constant') >= 0.5, 1, 0)
	return row

train_img_df['images'] = train_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])
train_img_df['images'] = train_img_df['images'].map(process_input)
av_int = train_img_df['images'].map(lambda x: x.mean())
print(av_int.mean())
print(av_int.std())
print(av_int.describe())
print(av_int.hist(bins=50))
plt.show()

train_img_df['images'] = train_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])
# train_img_df['images'] = train_img_df['images'].map(process_input).map(lambda x: 255-x if x.mean()>60 else x)
train_img_df['masks'] = train_img_df['masks'].map(read_and_stack_and_resize).map(lambda x: x.astype(int))
train_img_df = train_img_df.apply(resize_imgs, axis=1)
train_img_df['masks_resized'] = None
train_img_df = train_img_df.apply(resize_masks, axis=1)
train_img_df.sample(1)

# Show a few images
# Here we show a few images of the cells where we see there is a mixture of brightfield and fluorescence which will probably make using a single segmentation algorithm difficult

print('show a few images...')
n_img = 3
for i in range(0,10):	
	fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))
	print(m_axs)
	for (_, d_row), (c_im, c_lee, c_lab) in zip(train_img_df.sample(n_img).iterrows(), m_axs):
	    c_im.imshow(d_row['images'])
	    c_im.axis('off')
	    c_im.set_title('microscope')
	    
	    c_lee.imshow(d_row['masks'])
	    c_lee.axis('off')
	    c_lee.set_title('masks')

	    c_lab.imshow(d_row['masks_resized'])
	    c_lab.axis('off')
	    c_lab.set_title('masks_resized')
	    
	plt.show()


# fig, m_axs = plt.subplots(2, n_img, figsize = (12, 4))
# for i in range(0,10):
# 	for (_, c_row), (c_im, c_lab) in zip(train_img_df.sample(n_img).iterrows(), 
# 	                                     m_axs.T):
# 	    c_im.imshow(np.reshape(c_row['images'], newshape=c_row['images'].shape[0:2]))
# 	    c_im.axis('off')
# 	    c_im.set_title('Microscope')

# 	    c_im.imshow(c_row['masks'])
# 	    c_im.axis('off')
# 	    c_im.set_title('Labeled')
# 	    plt.show()
# sys.exit()
    # c_im.imshow(c_row['images'])
    # c_im.axis('off')
    # c_im.set_title('Microscope')
    
    # c_lab.imshow(c_row['masks'])
    # c_lab.axis('off')
    # c_lab.set_title('Labeled')

# Look at the intensity distribution
# Here we look briefly at the distribution of intensity and see a few groups forming, they should probably be handled separately. 
# print('look at intensity distribution...')
# train_img_df['Red'] = train_img_df['images'].map(lambda x: np.mean(x[:,:,0]))
# train_img_df['Green'] = train_img_df['images'].map(lambda x: np.mean(x[:,:,1]))
# train_img_df['Blue'] = train_img_df['images'].map(lambda x: np.mean(x[:,:,2]))
# train_img_df['Gray'] = train_img_df['images'].map(lambda x: np.mean(x))
# train_img_df['Red-Blue'] = train_img_df['images'].map(lambda x: np.mean(x[:,:,0]-x[:,:,2]))
# sns.pairplot(train_img_df[['Gray', 'Red', 'Green', 'Blue', 'Red-Blue']])

# Check Dimensions 
# Here we show the dimensions of the data to see the variety in the input images

print(train_img_df['images'].map(lambda x: x.shape).value_counts())

## Making a simple CNN
# Here we make a very simple CNN just to get a quick idea of how well it works. For this we use a batch normalization to normalize the inputs. We cheat a bit with the padding to keep problems simple.
# Simple CNN
# from keras.models import Sequential
# from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda
# simple_cnn = Sequential()
# simple_cnn.add(BatchNormalization(input_shape = (None, None, IMG_CHANNELS), 
#                                   name = 'NormalizeInput'))
# simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
# simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
# # use dilations to get a slightly larger field of view
# simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
# simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
# simple_cnn.add(Conv2D(32, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))

# # the final processing
# simple_cnn.add(Conv2D(16, kernel_size = (1,1), padding = 'same'))
# simple_cnn.add(Conv2D(1, kernel_size = (1,1), padding = 'same', activation = 'sigmoid'))
# simple_cnn.summary()
print('\nbuilding UNET...')
# UNET from https://www.kaggle.com/toregil/a-bigger-lung-net/notebook
# Larger UNET at https://github.com/zhixuhao/unet/blob/master/unet.py
from keras.layers import * #BatchNormalization, Conv2D, UpSampling2D, Lambda, MaxPool2D, Input
from keras.models import Model, Sequential

input_layer = Input(shape=(None,None,IMG_CHANNELS))

l = BatchNormalization()(input_layer)
c1 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c4)
c5 = Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(c5), c4], axis=-1)
l = Conv2D(filters=128, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c3], axis=-1)
l = Conv2D(filters=96, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
l = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
l = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')(l)
l = Dropout(0.25)(l)
l = Conv2D(filters=128, kernel_size=(1,1), activation='relu')(l)
l = Dropout(0.25)(l)
output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
                                                         
model = Model(input_layer, output_layer)

model.summary()

# input_layer = Input(shape=(None, None, 1,))
# conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
# # print()"conv1 shape:",conv1.shape)
# conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
# # print "conv1 shape:",conv1.shape
# pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
# # print "pool1 shape:",pool1.shape

# conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
# # print "conv2 shape:",conv2.shape
# conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
# # print "conv2 shape:",conv2.shape
# pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
# # print "pool2 shape:",pool2.shape

# conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
# # print "conv3 shape:",conv3.shape
# conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
# # print "conv3 shape:",conv3.shape
# pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
# # print "pool3 shape:",pool3.shape

# conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
# conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
# drop4 = Dropout(0.5)(conv4)
# pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

# conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
# conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
# drop5 = Dropout(0.5)(conv5)

# up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
# merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
# conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
# conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

# up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
# merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
# conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
# conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

# up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
# merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
# conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
# conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

# up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
# merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
# conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
# conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
# conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
# conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

# model = Model(input = inputs, output = conv10)

# model.summary()
# Loss
# Since we are being evaulated with intersection over union we can use the inverse of the DICE score as the loss function to optimize

from keras import backend as K
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
model.compile(optimizer = 'adam', 
                   loss = dice_coef_loss, 
                   metrics = [dice_coef, 'acc', 'mse'])

# Simple Training
# Here we run a simple training, with each image being it's own batch (not a very good idea), but it keeps the code simple

def simple_gen():
    while True:
        for _, c_row in train_img_df.iterrows():
            yield np.expand_dims(c_row['images'],0), np.expand_dims(np.expand_dims(c_row['masks'],-1),0)

model.fit_generator(simple_gen(), 
                         steps_per_epoch=train_img_df.shape[0],
                        epochs = 3)

# Apply Model to Test
# Here we apply the model to the test data

# %%time
test_df = img_df.query('TrainingSplit=="test"')
test_rows = []
group_cols = ['Stage', 'ImageId']
for n_group, n_rows in test_df.groupby(group_cols):
    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()
    test_rows += [c_row]
test_img_df = pd.DataFrame(test_rows)    
test_img_df['images'] = test_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])
# test_img_df['images'] = test_img_df['images'].map(process_input).map(lambda x: 255-x if x.mean()>60 else x)
test_img_df = test_img_df.apply(resize_imgs, axis=1)

print(test_img_df.shape[0], 'images to process')
test_img_df.sample(1)

# %%time
test_img_df['masks'] = test_img_df['images'].map(lambda x: model.predict(np.expand_dims(x, 0))[0, :, :, 0])

print(test_img_df.head())
test_img_df = test_img_df.apply(resize_masks, axis=1)
## Show a few predictions

n_img = 3
from skimage.morphology import closing, opening, disk
def clean_img(x):
    return opening(closing(x, disk(1)), disk(3))
for i in range(0,10):	
	fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))
	for (_, d_row), (c_im, c_lab, c_clean) in zip(test_img_df.sample(n_img).iterrows(), m_axs):
	    c_im.imshow(d_row['images'])
	    c_im.axis('off')
	    c_im.set_title('Microscope')
	    
	    c_lab.imshow(d_row['masks'])
	    c_lab.axis('off')
	    c_lab.set_title('Predicted Raw')
	    
	    c_clean.imshow(clean_img(d_row['masks_resized']))
	    c_clean.axis('off')
	    c_clean.set_title('Clean')
	plt.show()

# Check RLE
# Check that our approach for RLE encoding (stolen from [here](https://www.kaggle.com/rakhlin/fast-run-length-encoding-python)) works

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

## Calculate the RLEs for a Train Image

_, train_rle_row = next(train_img_df.tail(5).iterrows()) 
train_row_rles = list(prob_to_rles(train_rle_row['masks_resized']))

## Take the RLEs from the CSV

tl_rles = train_labels.query('ImageId=="{ImageId}"'.format(**train_rle_row))['EncodedPixels']

## Check
# Since we made some simplifications, we don't expect everything to be perfect, but pretty close

match, mismatch = 0, 0
for img_rle, train_rle in zip(sorted(train_row_rles, key = lambda x: x[0]), 
                             sorted(tl_rles, key = lambda x: x[0])):
    for i_x, i_y in zip(img_rle, train_rle):
        if i_x == i_y:
            match += 1
        else:
            mismatch += 1
print('Matches: %d, Mismatches: %d, Accuracy: %2.1f%%' % (match, mismatch, 100.0*match/(match+mismatch)))

# Calculate RLE for all the masks
# Here we generate the RLE for all the masks and output the the results to a table. We use a few morphological operations to clean up the images before submission since they can be very messy (remove single pixels, connect nearby regions, etc)

test_img_df['rles'] = test_img_df['masks_resized'].map(clean_img).map(lambda x: list(prob_to_rles(x)))
print(test_img_df.head())
out_pred_list = []
for _, c_row in test_img_df.iterrows():
    for c_rle in c_row['rles']:
        out_pred_list+=[dict(ImageId=c_row['ImageId'], 
                             EncodedPixels = ' '.join(np.array(c_rle).astype(str)))]
out_pred_df = pd.DataFrame(out_pred_list)
print(out_pred_df.shape[0], 'regions found for', test_img_df.shape[0], 'images')
out_pred_df.sample(3)

out_pred_df[['ImageId', 'EncodedPixels']].to_csv('/Users/wep/Kaggle/bowl/output/predictions.csv', index = False)
