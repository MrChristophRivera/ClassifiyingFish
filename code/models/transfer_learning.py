# Transfer learning with vgg16
# This code is based on the tutorial at
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
from os import getcwd
from os.path import join, split

import numpy as np
from keras.applications import vgg16
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# from keras import backend as K
# from keras.utils.np_utils import convert_kernel
# import tensorflow as tf


def save_bottleneck_features(train_data_dir, validation_data_dir, classes=None, save_directory=None, img_width=256,
                             img_height=256,
                             nb_train_samples=32, nb_validation_samples=32):
    """Predicts with the convolution layers of VGG16 and saves the output
    Args:
        train_data_dir(str): path to training data
        validation_data_dir(str): path to validation data
        save_directory(str): path to save the files
        classes(list): list of class labels
        img_width(int)
        img_height(int)
        nb_train_samples(int): number of training samples
        nb_validation_samples(int):
    Returns:
        None
    """

    # load the vgg16 covolutional layers
    model = vgg16.VGG16(include_top=False, input_shape=(img_width, img_height, 3))

    if classes is None:
        classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    # set up a data generator for the training data
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        train_data_dir,
        classes=classes,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)


    # predict on the training data
    bottleneck_features_train = model.predict_generator(generator=generator, val_samples=nb_train_samples)
    file_name = join(save_directory, 'bottleneck_features_train.npy')
    np.save(open(file_name, 'w'), bottleneck_features_train)

    # set up a generator for the validation_data
    generator = datagen.flow_from_directory(
        validation_data_dir,
        classes=classes,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    # predict on the validation data
    bottleneck_features_validation = model.predict_generator(generator=generator, val_samples=nb_validation_samples)
    file_name = join(save_directory, 'bottleneck_features_validation.npy')
    np.save(open(file_name, 'w'), bottleneck_features_validation)

    return None


def softmax(input_shape, nb_classes):
    """sets up a soft max with number of layers
    Args:
        input_shape(int): the shape for the intput
        nb_classes(int): number of classes
    Returns
        softmax(keras.models.Sequential)
    """

    # set up the
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def train_top_model(training_features_path, validation_feautures_path, top_model_weights_path, nb_classes=8,
                    nb_epochs=50,
                    batch_size=32):
    """ Trains a top softmax model with categorical crossentropy as the optimization function and save the weights.
        This is to initialize the weights
    Args:
        training_features_path(str): the path to the training features
        validation_feautures_path(str): the path to the validation features
        top_model_weights_path(str): path for saving the top model
        nb_classes(int): the number of classes for each
        train_labels(list): a list of integers for the labels
        validation_labels(list): a list of of integers for the validadtion labels
        nb_epochs(int): number of epochs to train
        batch_size(int): the number of samples per batch
    Returns:
        the top model
    """

    # load the training and validatoin data
    train_data = np.load(open(training_features_path))
    validation_data = np.load(open(validation_feautures_path))

    # set up the softmax top model
    model = softmax(input_shape=train_data.shape[1:], nb_classes=nb_classes)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose=0)

    model.save_weights(top_model_weights_path)
    return model


def init_top_model_pipeline():
    """A pipeline for initializing the weights of the top model
    """

    # classes
    classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    base = join(split(split(getcwd())[0])[0], 'data_local')
    train_dir = join(base, 'train')
    validation_dir = join(base, 'validation')
    save_directory = join(base, 'model_weights')

    # set up the values for training and
    number_train = 32
    number_validation = 32

    # save the bottle neck features
    save_bottleneck_features(train_data_dir=train_dir, validation_data_dir=validation_dir,
                             save_directory= save_directory,nb_train_samples=number_train,
                             nb_validation_samples=number_validation)

    top_model_weights_path = join(base, 'bottleneck_fc_model.h5')
    # train the top model
    train_features = join(save_directory, 'bottleneck_features_train.npy')
    validation_features = join(save_directory, 'bottleneck_features_validation.npy')
    top_model_path = join(save_directory, 'top_model_weights.h5')
    #train_top_model(training_features_path = train_features, validation_feautures_path = validation_features,
                    #top_model_weights_path= top_model_weights_path, nb_classes=8, nb_epochs=10,batch_size=32)


if __name__ == '__main__':
    g = init_top_model_pipeline()

