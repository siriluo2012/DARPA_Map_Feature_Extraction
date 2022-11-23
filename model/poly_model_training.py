import os
import random
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np 
from sklearn.model_selection import train_test_split

# for bulding and running deep learning model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator


def load_train_img(filename):
    
    # filename = ('CO_Boulder_1978_Ysp_poly_18_7.png', 'CO_Boulder_1978_Ysp_poly.png')

    mapName = '/home/shared/DARPA/all_patched_data/training/poly/map_patches/'+filename[0]
    legendName = '/home/shared/DARPA/all_patched_data/training/poly/legend/'+filename[1] 

    map_img = tf.io.read_file(mapName) # Read image file
    map_img = tf.cast(tf.io.decode_png(map_img), dtype=tf.float32) / 255.0

    legend_img = tf.io.read_file(legendName) # Read image file
    legend_img = tf.cast(tf.io.decode_png(legend_img), dtype=tf.float32) / 255.0
    
    map_img = tf.concat(axis=2, values = [map_img, legend_img])
    map_img = data_augmentation(map_img)
    
    map_img = map_img*2.0 - 1.0 # range(-1.0,1.0)
    map_img = tf.image.resize(map_img, [256, 256])
    
    segName = '/home/shared/DARPA/all_patched_data/training/poly/seg_patches/'+filename[0]  
    
    legend_img = tf.io.read_file(segName) # Read image file
    legend_img = tf.io.decode_png(legend_img)
    legend_img = tf.image.resize(legend_img, [256, 256])
    
    return map_img, legend_img

# img, seg = load_img(('CO_Boulder_1978_Ysp_poly_18_7.png', 'CO_Boulder_1978_Ysp_poly.png'))

def load_validation_img(filename):
    
    # filename = ('CO_Boulder_1978_Ysp_poly_18_7.png', 'CO_Boulder_1978_Ysp_poly.png')

    mapName = '/home/shared/DARPA/all_patched_data/validation/poly/map_patches/'+filename[0]
    legendName = '/home/shared/DARPA/all_patched_data/validation/poly/legend/'+filename[1] 

    map_img = tf.io.read_file(mapName) # Read image file
    map_img = tf.cast(tf.io.decode_png(map_img), dtype=tf.float32) / 255.0

    legend_img = tf.io.read_file(legendName) # Read image file
    legend_img = tf.cast(tf.io.decode_png(legend_img), dtype=tf.float32) / 255.0
    
    map_img = tf.concat(axis=2, values = [map_img, legend_img])
    map_img = map_img*2.0 - 1.0 # range(-1.0,1.0)
    map_img = tf.image.resize(map_img, [256, 256])
    
    segName = '/home/shared/DARPA/all_patched_data/validation/poly/seg_patches/'+filename[0]  
    
    legend_img = tf.io.read_file(segName) # Read image file
    legend_img = tf.io.decode_png(legend_img)
    legend_img = tf.image.resize(legend_img, [256, 256])
    legend_img = legend_img
    
    return map_img, legend_img

# img, seg = load_img(('CO_Boulder_1978_Ysp_poly_18_7.png', 'CO_Boulder_1978_Ysp_poly.png'))

train_map_file = os.listdir('/home/shared/DARPA/all_patched_data/training/poly/map_patches')
random.shuffle(train_map_file)
train_map_legend_names = [(x, '_'.join(x.split('_')[0:-2])+'.png') for x in train_map_file]

train_dataset = tf.data.Dataset.from_tensor_slices(train_map_legend_names)
train_dataset = train_dataset.map(load_train_img)
train_dataset = train_dataset.shuffle(5000, reshuffle_each_iteration=False).batch(128)

# A peek of how BatchDataset 
# it = iter(train_dataset)
# print(next(it))
validate_map_file = os.listdir('/home/shared/DARPA/all_patched_data/validation/poly/map_patches')
validate_map_legend_names = [(x, '_'.join(x.split('_')[0:-2])+'.png') for x in validate_map_file]

validate_dataset = tf.data.Dataset.from_tensor_slices(validate_map_legend_names)
validate_dataset = validate_dataset.map(load_validation_img)
validate_dataset = validate_dataset.batch(50)

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    
    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv
    
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv

def UNetCompiled(input_size=(256, 256, 6), n_filters=32, n_classes=1):
    """
       Combine both encoder and decoder blocks according to the U-Net research paper
       Return the model as output 
    """
    # Input size represent the size of 1 image (the size used for pre-processing) 
    inputs = Input(input_size)
    
    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
    cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same', activation="sigmoid")(conv9)
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

# Call the helper function for defining the layers for the model, given the input image size
unet = UNetCompiled(input_size=(256,256,6), n_filters=16, n_classes=1)

unet.compile(optimizer=tf.keras.optimizers.Adam(), 
             loss=tf.keras.losses.binary_crossentropy, #SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', 'acc'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./saved_model/best_train_validation_model_16_filter.hdf5', 
    monitor='loss',
    verbose=0, 
    save_best_only=True,
    save_freq= 200)

#load weights
if os.path.exists("./saved_model/saved/saved_best_train_validation_model_16_filter.hdf5"):
    unet.load_weights("./saved_model/saved/saved_best_train_validation_model_16_filter.hdf5")

# Run the model in a mini-batch fashion and compute the progress for each epoch
results = unet.fit(train_dataset, epochs=5, callbacks=[cp_callback], validation_data=validate_dataset)

# serialize and save the model that you just trained 
#saved_model_path = "/home/shirui/DARPA/DARPAMapExtraction/model/saved_model/my_model.h5" 
saved_model_path = "./saved_model/best_train_validation_model_16_filter.hdf5"
unet.save(saved_model_path)

