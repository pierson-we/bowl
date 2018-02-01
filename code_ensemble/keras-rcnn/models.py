from __future__ import print_function
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda


def Unet(model_input, num_classes):
    s = Lambda(lambda x: x / 255)(model_input)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    return outputs
    
    # model = Model(inputs=[model_input], outputs=[outputs], name='Unet')

    # return model

def DeepLab(model_input, num_classes):

    import theano.tensor as T

    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, Conv2D, MaxPooling2D, merge, ZeroPadding2D, Dropout, UpSampling2D
    from keras.layers.core import Layer, InputSpec
    from keras.utils.layer_utils import convert_all_kernels_in_model
    from keras.utils.data_utils import get_file
    from keras.engine.topology import get_source_inputs
    import numpy as np
    from keras.layers import Activation, Permute, Reshape

    def softmax(x, restore_shape=True):
        """
        Softmax activation for a tensor x. No need to unroll the input first.
        :param x: x is a tensor with shape (None, channels, h, w)
        :param restore_shape: if False, output is returned unrolled (None, h * w, channels)
        :return: softmax activation of tensor x
        """
        _, c, h, w = x._keras_shape
        x = Permute(dims=(2, 3, 1))(x)
        x = Reshape(target_shape=(h * w, c))(x)

        x = Activation('softmax')(x)

        if restore_shape:
            x = Reshape(target_shape=(h, w, c))(x)
            x = Permute(dims=(3, 1, 2))(x)

        return x

    def DeeplabV2(model_input, upsampling=8, apply_softmax=True,
                  input_tensor=None,
                  classes=num_classes):
        """Instantiate the DeeplabV2 architecture with VGG16 encoder,
        optionally loading weights pre-trained on segmentation.
        Note that pre-trained model is only available for Theano dim ordering.
        The model and the weights should be compatible with both
        TensorFlow and Theano backends.
        # Arguments
            input_shape: shape tuple. It should have exactly 3 inputs channels,
                and the axis ordering should be coherent with what specified in
                your keras.json (e.g. use (3, 512, 512) for 'th' and (512, 512, 3)
                for 'tf').
            upsampling: final front end upsampling (default is 8x).
            apply_softmax: whether to apply softmax or return logits.
            weights: one of `None` (random initialization)
                or `voc2012` (pre-training on VOC2012 segmentation).
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        """
        from keras.layers import Activation

        
        # img_input = Input(shape=(img_size, img_size, img_channels))

        # Block 1
        h = ZeroPadding2D(padding=(1, 1))(model_input)
        h = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

        # Block 2
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

        # Block 3
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

        # Block 4
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

        # Block 5
        h = ZeroPadding2D(padding=(2, 2))(h)
        h = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1')(h)
        h = ZeroPadding2D(padding=(2, 2))(h)
        h = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2')(h)
        h = ZeroPadding2D(padding=(2, 2))(h)
        h = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

        # branching for Atrous Spatial Pyramid Pooling
        # hole = 6
        b1 = ZeroPadding2D(padding=(6, 6))(p5)
        b1 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', name='fc6_1')(b1)
        b1 = Dropout(0.5)(b1)
        b1 = Conv2D(1024, (1, 1), activation='relu', name='fc7_1')(b1)
        b1 = Dropout(0.5)(b1)
        b1 = Conv2D(classes, (1, 1), activation='relu', name='fc8_voc12_1')(b1)

        # hole = 12
        b2 = ZeroPadding2D(padding=(12, 12))(p5)
        b2 = Conv2D(1024, (3, 3), dilation_rate=(12, 12), activation='relu', name='fc6_2')(b2)
        b2 = Dropout(0.5)(b2)
        b2 = Conv2D(1024, (1, 1), activation='relu', name='fc7_2')(b2)
        b2 = Dropout(0.5)(b2)
        b2 = Conv2D(classes, (1, 1), activation='relu', name='fc8_voc12_2')(b2)

        # hole = 18
        b3 = ZeroPadding2D(padding=(18, 18))(p5)
        b3 = Conv2D(1024, (3, 3), dilation_rate=(18, 18), activation='relu', name='fc6_3')(b3)
        b3 = Dropout(0.5)(b3)
        b3 = Conv2D(1024, (1, 1), activation='relu', name='fc7_3')(b3)
        b3 = Dropout(0.5)(b3)
        b3 = Conv2D(classes, (1, 1), activation='relu', name='fc8_voc12_3')(b3)

        # hole = 24
        b4 = ZeroPadding2D(padding=(24, 24))(p5)
        b4 = Conv2D(1024, (3, 3), dilation_rate=(24, 24), activation='relu', name='fc6_4')(b4)
        b4 = Dropout(0.5)(b4)
        b4 = Conv2D(1024, (1, 1), activation='relu', name='fc7_4')(b4)
        b4 = Dropout(0.5)(b4)
        b4 = Conv2D(classes, (1, 1), activation='relu', name='fc8_voc12_4')(b4)

        s = merge([b1, b2, b3, b4], mode='sum')
        logits = UpSampling2D(size=upsampling)(s)

        out = Conv2D(1, (1,1), activation='sigmoid')(logits)
        # out = Activation('sigmoid')(logits)
        # out = softmax(logits)
        # inputs = img_input

        return out
        # Create model.
        # model = Model(model_input, out, name='DeepLabV2')

        # return model
    return DeeplabV2(model_input, classes=num_classes)


# from https://github.com/Borda/segment-nuclei/blob/master/code/helper/model_builder.py

def random_masters(model_input, num_classes):
    import keras.layers
    import keras.models
    import tensorflow as tf

    # hyperparameters TODO maybe move to main file?
    FLAG_BN = True
    FLAG_DO = False
    FLAG_DO_LAST_LAYER = False
    CONST_DO_RATE = 0.3



    option_dict_conv = {"activation": "relu", "border_mode": "same"}
    option_dict_bn = {"mode": 0, "momentum" : 0.9}


    # returns a core model from gray input to 64 channels of the same size
    def get_core(model_input):
        
        # x = keras.layers.Input(shape=(dim1, dim2, 3))

        # DOWN

        a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(model_input)  
        # a = tf.Print(a, [x], summarize = 100)
        if FLAG_BN:
            a = keras.layers.BatchNormalization(**option_dict_bn)(a)
        if FLAG_DO:
            a = keras.layers.Dropout(CONST_DO_RATE)(a)

        a = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(a)
        if FLAG_BN:
            a = keras.layers.BatchNormalization(**option_dict_bn)(a)
        if FLAG_DO:
            a = keras.layers.Dropout(CONST_DO_RATE)(a)

        y = keras.layers.MaxPooling2D()(a)

        b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
        if FLAG_BN:
            b = keras.layers.BatchNormalization(**option_dict_bn)(b)
        if FLAG_DO:
            b = keras.layers.Dropout(CONST_DO_RATE)(b)

        b = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(b)
        if FLAG_BN:
            b = keras.layers.BatchNormalization(**option_dict_bn)(b)
        if FLAG_DO:
            b = keras.layers.Dropout(CONST_DO_RATE)(b)

        y = keras.layers.MaxPooling2D()(b)

        c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
        if FLAG_BN:
            c = keras.layers.BatchNormalization(**option_dict_bn)(c)
        if FLAG_DO:
            c = keras.layers.Dropout(CONST_DO_RATE)(c)

        c = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(c)
        if FLAG_BN:
            c = keras.layers.BatchNormalization(**option_dict_bn)(c)
        if FLAG_DO:
            c = keras.layers.Dropout(CONST_DO_RATE)(c)

        y = keras.layers.MaxPooling2D()(c)

        d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(y)
        if FLAG_BN:
            d = keras.layers.BatchNormalization(**option_dict_bn)(d)
        if FLAG_DO:
            d = keras.layers.Dropout(CONST_DO_RATE)(d)

        d = keras.layers.Convolution2D(512, 3, 3, **option_dict_conv)(d)
        if FLAG_BN:
            d = keras.layers.BatchNormalization(**option_dict_bn)(d)
        if FLAG_DO:
            d = keras.layers.Dropout(CONST_DO_RATE)(d)

        # UP

        d = keras.layers.UpSampling2D()(d)

        y = keras.layers.merge([d, c], concat_axis=3, mode="concat")

        e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(y)
        if FLAG_BN:
            e = keras.layers.BatchNormalization(**option_dict_bn)(e)
        if FLAG_DO:
            e = keras.layers.Dropout(CONST_DO_RATE)(e)

        e = keras.layers.Convolution2D(256, 3, 3, **option_dict_conv)(e)
        if FLAG_BN:
            e = keras.layers.BatchNormalization(**option_dict_bn)(e)
        if FLAG_DO:
            e = keras.layers.Dropout(CONST_DO_RATE)(e)

        e = keras.layers.UpSampling2D()(e)

        y = keras.layers.merge([e, b], concat_axis=3, mode="concat")

        f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(y)
        if FLAG_BN:
            f = keras.layers.BatchNormalization(**option_dict_bn)(f)
        if FLAG_DO:
            f = keras.layers.Dropout(CONST_DO_RATE)(f)

        f = keras.layers.Convolution2D(128, 3, 3, **option_dict_conv)(f)
        if FLAG_BN:
            f = keras.layers.BatchNormalization(**option_dict_bn)(f)
        if FLAG_DO:
            f = keras.layers.Dropout(CONST_DO_RATE)(f)

        f = keras.layers.UpSampling2D()(f)

        y = keras.layers.merge([f, a], concat_axis=3, mode="concat")

        # HEAD

        y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
        if FLAG_BN:
            y = keras.layers.BatchNormalization(**option_dict_bn)(y)
        if FLAG_DO:
            y = keras.layers.Dropout(CONST_DO_RATE)(y)

        y = keras.layers.Convolution2D(64, 3, 3, **option_dict_conv)(y)
        if FLAG_BN:
            y = keras.layers.BatchNormalization(**option_dict_bn)(y)
        if FLAG_DO_LAST_LAYER:
            y = keras.layers.Dropout(CONST_DO_RATE)(y)
        
        return y


    def get_model_1_class(model_input, num_classes):
    
        y = get_core(model_input)
        
        y = keras.layers.Convolution2D(num_classes, 1, padding="same", activation="sigmoid")(y)
        # y = keras.layers.Convolution2D(num_classes, (1, 1), **option_dict_conv)(y)
        
        # y = keras.layers.Activation("sigmoid")(y)

        return y
        # model = keras.models.Model(inputs=model_input, outputs=y)

        # return model

    return get_model_1_class(model_input, num_classes)

# from https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow/blob/master/layers_builder.py

def PSPnet(model_input, num_classes, img_size):
    from math import ceil
    from keras import layers
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
    from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
    from keras.layers.merge import Concatenate, Add
    from keras.models import Model
    from keras.optimizers import SGD
    from keras.backend import tf as ktf
    from keras import backend as K

    import tensorflow as tf

    learning_rate = 1e-3  # Layer specific learning rate
    # Weight decay not implemented


    def BN(name=""):
        return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


    class Interp(layers.Layer):

        def __init__(self, new_size, **kwargs):
            self.new_size = new_size
            super(Interp, self).__init__(**kwargs)

        def build(self, input_shape):
            super(Interp, self).build(input_shape)

        def call(self, inputs, **kwargs):
            new_height, new_width = self.new_size
            resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                              align_corners=True)
            return resized

        def compute_output_shape(self, input_shape):
            return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

        def get_config(self):
            config = super(Interp, self).get_config()
            config['new_size'] = self.new_size
            return config


    # def Interp(x, shape):
    #    new_height, new_width = shape
    #    resized = ktf.image.resize_images(x, [new_height, new_width],
    #                                      align_corners=True)
    #    return resized


    def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
        lvl = str(lvl)
        sub_lvl = str(sub_lvl)
        names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
                 "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
                 "conv" + lvl + "_" + sub_lvl + "_3x3",
                 "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
                 "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
                 "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
        if modify_stride is False:
            prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
                          use_bias=False)(prev)
        elif modify_stride is True:
            prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
                          use_bias=False)(prev)

        prev = BN(name=names[1])(prev)
        prev = Activation('relu')(prev)

        prev = ZeroPadding2D(padding=(pad, pad))(prev)
        prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                      name=names[2], use_bias=False)(prev)

        prev = BN(name=names[3])(prev)
        prev = Activation('relu')(prev)
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
                      use_bias=False)(prev)
        prev = BN(name=names[5])(prev)
        return prev


    def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
        lvl = str(lvl)
        sub_lvl = str(sub_lvl)
        names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
                 "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

        if modify_stride is False:
            prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
                          use_bias=False)(prev)
        elif modify_stride is True:
            prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
                          use_bias=False)(prev)

        prev = BN(name=names[1])(prev)
        return prev


    def empty_branch(prev):
        return prev


    def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
        prev_layer = Activation('relu')(prev_layer)
        block_1 = residual_conv(prev_layer, level,
                                pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                                modify_stride=modify_stride)

        block_2 = short_convolution_branch(prev_layer, level,
                                           lvl=lvl, sub_lvl=sub_lvl,
                                           modify_stride=modify_stride)
        added = Add()([block_1, block_2])
        return added


    def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
        prev_layer = Activation('relu')(prev_layer)

        block_1 = residual_conv(prev_layer, level, pad=pad,
                                lvl=lvl, sub_lvl=sub_lvl)
        block_2 = empty_branch(prev_layer)
        added = Add()([block_1, block_2])
        return added


    def ResNet(inp, layers):
        # Names for the first couple layers of model
        names = ["conv1_1_3x3_s2",
                 "conv1_1_3x3_s2_bn",
                 "conv1_2_3x3",
                 "conv1_2_3x3_bn",
                 "conv1_3_3x3",
                 "conv1_3_3x3_bn"]

        # Short branch(only start of network)

        cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                      use_bias=False)(inp)  # "conv1_1_3x3_s2"
        bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
        relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

        cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                      use_bias=False)(relu1)  # "conv1_2_3x3"
        bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
        relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

        cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                      use_bias=False)(relu1)  # "conv1_3_3x3"
        bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
        relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

        res = MaxPooling2D(pool_size=(3, 3), padding='same',
                           strides=(2, 2))(relu1)  # "pool1_3x3_s2"

        # ---Residual layers(body of network)

        """
        Modify_stride --Used only once in first 3_1 convolutions block.
        changes stride of first convolution from 1 -> 2
        """

        # 2_1- 2_3
        res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
        for i in range(2):
            res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

        # 3_1 - 3_3
        res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
        for i in range(3):
            res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)
        if layers is 50:
            # 4_1 - 4_6
            res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
            for i in range(5):
                res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
        elif layers is 101:
            # 4_1 - 4_23
            res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
            for i in range(22):
                res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
        else:
            print("This ResNet is not implemented")

        # 5_1 - 5_3
        res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
        for i in range(2):
            res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

        res = Activation('relu')(res)
        return res


    def interp_block(prev_layer, level, feature_map_shape, input_shape):
        if input_shape == (512, 512):
            kernel_strides_map = {1: 60,
                                  2: 30,
                                  3: 20,
                                  6: 10}
        elif input_shape == (713, 713):
            kernel_strides_map = {1: 90,
                                  2: 45,
                                  3: 30,
                                  6: 15}
        elif input_shape == (256, 256):
            kernel_strides_map = {1: 32,
                                  2: 16,
                                  3: 10,
                                  6: 5}
        elif input_shape == (128, 128):
            kernel_strides_map = {1: 16,
                                  2: 8,
                                  3: 5,
                                  6: 2}
        else:
            print("Pooling parameters for input shape ",
                  input_shape, " are not defined.")
            exit(1)

        names = [
            "conv5_3_pool" + str(level) + "_conv",
            "conv5_3_pool" + str(level) + "_conv_bn"
        ]
        kernel = (kernel_strides_map[level], kernel_strides_map[level])
        strides = (kernel_strides_map[level], kernel_strides_map[level])
        prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
        prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                            use_bias=False)(prev_layer)
        prev_layer = BN(name=names[1])(prev_layer)
        prev_layer = Activation('relu')(prev_layer)
        # prev_layer = Lambda(Interp, arguments={
        #                    'shape': feature_map_shape})(prev_layer)
        prev_layer = Interp(feature_map_shape)(prev_layer)
        return prev_layer


    def build_pyramid_pooling_module(res, input_shape):
        """Build the Pyramid Pooling Module."""
        # ---PSPNet concat layers with Interpolation
        feature_map_size = tuple(int(ceil(input_dim / 8.0))
                                 for input_dim in input_shape)
        print("PSP module will interpolate to a final feature map size of %s" %
              (feature_map_size, ))

        interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
        interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
        interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
        interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

        # concat all these layers. resulted
        # shape=(1,feature_map_size_x,feature_map_size_y,4096)
        res = Concatenate()([res,
                             interp_block6,
                             interp_block3,
                             interp_block2,
                             interp_block1])
        return res

    def depth_softmax(matrix):
        sigmoid = lambda x: 1 / (1 + K.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def build_pspnet(nb_classes, resnet_layers, input_shape):
        """Build PSPNet."""
        print("Building a PSPNet based on ResNet %i expecting inputs of shape %s predicting %i classes" % (
            resnet_layers, input_shape, nb_classes))

        # inp = Input((input_shape[0], input_shape[1], 3))
        res = ResNet(model_input, layers=resnet_layers)
        psp = build_pyramid_pooling_module(res, input_shape)

        x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
                   use_bias=False)(psp)
        x = BN(name="conv5_4_bn")(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
        # x = Lambda(Interp, arguments={'shape': (
        #    input_shape[0], input_shape[1])})(x)
        x = Interp([input_shape[0], input_shape[1]])(x)
        x = Activation('sigmoid')(x)
        # x = Lambda(depth_softmax)(x)

        return x
        # model = Model(inputs=model_input, outputs=x)
        # # model.summary()

        # # Solver
        
        return model
    return build_pspnet(nb_classes=num_classes, resnet_layers=50, input_shape=(img_size, img_size))

def simple_cnn(img_size, img_channels):
    from keras.models import Sequential
    from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda
    model = Sequential()
    model.add(BatchNormalization(input_shape = (None, None, img_channels), 
                                      name = 'NormalizeInput'))
    model.add(Conv2D(16, kernel_size = (3,3), padding = 'same'))
    model.add(Conv2D(16, kernel_size = (3,3), padding = 'same'))
    # use dilations to get a slightly larger field of view
    model.add(Conv2D(32, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
    # model.add(Dropout)
    model.add(Conv2D(32, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
    model.add(Conv2D(64, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))

    # the final processing
    model.add(Conv2D(16, kernel_size = (1,1), padding = 'same'))
    model.add(Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'sigmoid'))
    return model
