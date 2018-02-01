import sys
import os
sys.path.insert(0, os.path.join(os.pardir, 'packages'))
print(os.path.join(os.pardir, 'packages'))
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import transform
from keras.optimizers import SGD, RMSprop, Adam
from keras.losses import logcosh
from keras import callbacks
from keras.models import Model
from keras.layers import Input, Conv2D, concatenate, average, Dropout
import keras_rcnn
import matplotlib.pyplot as plt
import utils
import models
import losses


if __name__ == "__main__":
    img_size = 128
    img_channels = 3
    batch_size = 4
    train_path = '/home/paperspace/bowl/input/stage1_train/'
    test_path = '/home/paperspace/bowl/input/stage1_test/'
    model_path = '/home/paperspace/bowl/models/%sweights.{epoch:03d}-{val_loss:.4f}.hdf5' % img_size
    classes_csv = '/home/paperspace/bowl/input/classes.csv'

    # train_path = '/Users/wep/Kaggle/bowl/input/stage1_train/'
    # test_path = '/Users/wep/Kaggle/bowl/input/stage1_test/'
    # model_path = '/Users/wep/Kaggle/bowl/models/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    # classes_csv = '/Users/wep/Kaggle/bowl/input/classes.csv'
    
    model_input = Input((img_size, img_size, img_channels))

    model = keras_rcnn.models.RCNN(model_input, classes=num_classes + 1)
    
    X_train, Y_train, test_img_df, num_classes = utils.make_df(train_path, test_path, img_size, classes_csv)

    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)

    train_weights = utils.get_train_weights(xtr)

    train_generator, val_generator = utils.generator(xtr, xval, ytr, yval, batch_size)
    

    # sum_int = []
    # for i in range(0, X_train.shape[0]):
    #     sum_int.append(np.mean(X_train[i,:,:,:]))

    # sum_int = pd.Series(sum_int)
    # print(sum_int.mean())
    # print(sum_int.std())
    # print(sum_int.describe())
    # print(sum_int.hist(bins=100))
    # plt.show()

    # sys.exit()


    # row = 0
    # n_img = 4
    # for i in range(0, 3): # (test_img_df.shape[0]/(n_img*3) + 1)):   
    #     fig, m_axs = plt.subplots(nrows=2, ncols=n_img, figsize = (12, 6))
    #     for c_im1, c_im2, c_im3, c_im4 in m_axs:
    #         c_im1.imshow(X_train[row])
    #         c_im1.axis('off')
    #         c_im1.set_title('microscope')

    #         c_im2.imshow(Y_train[row][:,:,0])
    #         c_im2.axis('off')
    #         c_im2.set_title('mask1')

    #         c_im3.imshow(Y_train[row][:,:,1])
    #         c_im3.axis('off')
    #         c_im3.set_title('mask2')

    #         c_im4.imshow(Y_train[row][:,:,2])
    #         c_im4.axis('off')
    #         c_im4.set_title('mask3')

    #         row += 1
    #         if row == len(X_train): break

    #     plt.show()

    # n_img = 2
    # row = 0
    # for i in range(0, 10): # (test_img_df.shape[0]/(n_img*3) + 1)):   
    #     fig, m_axs = plt.subplots(4, n_img, figsize = (12, 6))
    #     for c_im1, c_im2 in m_axs: #, c_im3, c_im4, c_im5, c_im6 in m_axs:
    #         # print(test_img_df.iloc[row]['masks_multiclass'].shape)
    #         c_im1.imshow(Y_train[row][:,:,0])
    #         c_im1.axis('off')
    #         # c_im1.set_title('microscope')

    #         c_im2.imshow(Y_train[row][:,:,1])
    #         c_im2.axis('off')
    #         # c_im2.set_title('image raw')

    #         # c_im3.imshow(Y_train[row][:,:,2])
    #         # c_im3.axis('off')
    #         # # c_im3.set_title('mask')

    #         # c_im4.imshow(Y_train[row][:,:,3])
    #         # c_im4.axis('off')
    #         # # c_im3.set_title('mask')

    #         # c_im5.imshow(Y_train[row][:,:,4])
    #         # c_im5.axis('off')
    #         # # c_im3.set_title('mask')

    #         # c_im6.imshow(X_train[row])
    #         # c_im6.axis('off')
    #         # # c_im3.set_title('mask')

    #         row += 1
    #         if row == test_img_df.shape[0]: break

    #     plt.show()

    checkpoint = callbacks.ModelCheckpoint(model_path, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=10)
    LR = callbacks.LearningRateScheduler(utils.lr_schedule)

    adam = Adam(0.0001)

    model.compile(optimizer=adam) # , loss='binary_crossentropy', metrics=[losses.mean_iou, losses.dice_coef, losses.precision, 'acc'])

    model.fit(x=xtr, y=ytr, batch_size=batch_size, validation_data=(xval, yval), sample_weight=train_weights, epochs=1)

    # model.save_weights('/home/paperspace/bowl/models/ensemble_weights.hdf5')

    # model.summary()
    # model.load_weights('/home/paperspace/bowl/models/473weights.019-0.2950.hdf5')
    # model.load_weights('/Users/wep/Kaggle/bowl/models/473weights.019-0.2950.hdf5')
    # model.compile(optimizer='adam', loss=losses.bce_dice_loss, metrics=[losses.mean_iou, 'acc'])# loss=losses.bce_dice_loss, metrics=[losses.mean_iou, 'acc'])
    
    # model.compile(optimizer=sgd, # RMSprop(lr = 1e-4),
    #               loss=losses.bce_dice_loss,
    #               metrics=[losses.mean_iou, losses.dice_coef, losses.precision, 'acc'])

    # model.fit_generator(train_generator, steps_per_epoch=len(xtr)/(batch_size), epochs=4,
    #                     validation_data=val_generator, validation_steps=len(xval)/batch_size, callbacks=[checkpoint, LR]) # , LR])
    
    # test_img_df = test_img_df.iloc[5:10]
    # test_img_df.head()
    test_img_df['masks'] = test_img_df['images'].map(lambda x: model.predict(x)) # np.expand_dims(x, 0))[0, :, :, :])
    # test_img_df['masks'] = test_img_df['masks_multiclass'].map(utils.masks_1_class) # utils.masks_1_class)
    print(test_img_df['masks'].iloc[0].shape)

    # test_img_df['masks'] = test_img_df['masks_ensemble'].map(lambda x: np.mean(x, axis=0))
    # test_img_df['masks_bg'] = test_img_df['masks_multiclass'].map(lambda x: x[:,:,-1])
    
    # n_img = 3
    # row = 0
    # for i in range(0, 5): # (test_img_df.shape[0]/(n_img*3) + 1)):   
    #     fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))
    #     for c_im1, c_im2, c_im3 in m_axs:
    #         print(test_img_df.iloc[row]['masks_3_classes'].shape)
    #         c_im1.imshow(np.reshape(test_img_df.iloc[row]['masks_3_classes'][:,:,0], (img_size, img_size)))
    #         c_im1.axis('off')
    #         # c_im1.set_title('microscope')

    #         c_im2.imshow(np.reshape(test_img_df.iloc[row]['masks_3_classes'][:,:,1], (img_size, img_size)))
    #         c_im2.axis('off')
    #         # c_im2.set_title('image raw')

    #         c_im3.imshow(np.reshape(test_img_df.iloc[row]['masks_3_classes'][:,:,2], (img_size, img_size)))
    #         c_im3.axis('off')
    #         # c_im3.set_title('mask')

    #         row += 1
    #         if row == test_img_df.shape[0]: break

    #     plt.show()


    test_img_df, out_pred_df = utils.format_preds(Y_train, test_img_df)
    out_pred_df[['ImageId', 'EncodedPixels']].to_csv('/home/paperspace/bowl/output/predictions.csv', index = False)
    # out_pred_df[['ImageId', 'EncodedPixels']].to_csv('/Users/wep/Kaggle/bowl/output/predictions.csv', index = False)

    # n_img = 3
    # row = 0
    # for i in range(0,5): #(test_img_df.shape[0]/(n_img*3) + 1)):   
    #     fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))
    #     for c_im1, c_im2, c_im3 in m_axs:
    #         c_im1.imshow(test_img_df.iloc[row]['images_raw'])
    #         c_im1.axis('off')
    #         c_im1.set_title('microscope')

    #         c_im2.imshow(test_img_df.iloc[row]['masks'])
    #         c_im2.axis('off')
    #         c_im2.set_title('mask raw')

    #         c_im3.imshow(test_img_df.iloc[row]['masks_bg'])
    #         c_im3.axis('off')
    #         c_im3.set_title('mask bg')

    #         row += 1
    #         if row == test_img_df.shape[0]: break

    #     plt.show()

    n_img = 3
    row = 0
    for i in range(0, 22): # (test_img_df.shape[0]/(n_img*3) + 1)):   
        fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))
        for c_im1, c_im4, c_im5 in m_axs:
            # print(test_img_df.iloc[row]['masks_multiclass'].shape)
            c_im1.imshow(np.reshape(test_img_df.iloc[row]['masks'], (img_size, img_size)))
            c_im1.axis('off')
            # c_im1.set_title('microscope')

            # c_im2.imshow(np.reshape(test_img_df.iloc[row]['masks'][:,:,1], (img_size, img_size)))
            # c_im2.axis('off')
            # c_im2.set_title('image raw')

            # c_im3.imshow(np.reshape(test_img_df.iloc[row]['masks_multiclass'][:,:,2], (img_size, img_size)))
            # c_im3.axis('off')
            # # c_im3.set_title('mask')

            c_im4.imshow(np.reshape(utils.clean_img(test_img_df.iloc[row]['masks_resized']), (test_img_df.iloc[row]['shape'][0], test_img_df.iloc[row]['shape'][1])))
            c_im4.axis('off')
            # c_im6.set_title('microscope')

            c_im5.imshow(test_img_df.iloc[row]['images'])
            c_im5.axis('off')
            c_im5.set_title('Avg. intensity: %s' % np.mean(test_img_df.iloc[row]['images']))

            row += 1
            if row == test_img_df.shape[0]: break

        plt.show()

    
    # preds_test = model.predict(X_test, verbose=1)

    # preds_test_upsampled = []
    # for i in range(len(preds_test)):
    #     preds_test_upsampled.append(cv2.resize(preds_test[i], 
    #                                        (sizes_test[i][1], sizes_test[i][0])))
        
    # test_ids = next(os.walk(test_path))[1]
    # new_test_ids = []
    # rles = []
    # for n, id_ in enumerate(test_ids):
    #     rle = list(utils.prob_to_rles(preds_test_upsampled[n]))
    #     rles.extend(rle)
    #     new_test_ids.extend([id_] * len(rle))
    # sub = pd.DataFrame()
    # sub['ImageId'] = new_test_ids
    # sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    # sub.to_csv('/home/paperspace/bowl/output/sub.csv', index=False)
