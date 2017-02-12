# Transfer learning with vgg16

from os.path import join

import numpy as np
from keras.applications import vgg16
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# from keras import backend as K
# from keras.utils.np_utils import convert_kernel
# import tensorflow as tf


def save_bottleneck_feature(train_data_dir, validation_data_dir, save_directory, img_width=150, img_height=150,
                            nb_train_samples=100, nb_validation_samples=800):
    """Predicts with the convolution layers of VGG16 and saves the output
    Args:
        train_data_dir(str): path to training data
        validation_data_dir(str): path to validation data
        save_directory(str): path to save the files
        img_width(int)
        img_height(int)
        nb_train_samples(int): number of training samples
        nb_validation_samples(int):
    Returns:
        None
    """

    # load the vgg16 covolutional layers
    model = vgg16.VGG16(include_top=False, input_shape=(img_width, img_height, 3))

    # set up a data generator for the training data
    datagen = ImageDataGenerator(1. / 255)
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)

    # predict on the training data
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    file_name = join(save_directory, 'bottleneck_features_train.npy')
    np.save(open(file_name, 'w'), bottleneck_features_train)

    # set up a generator for the validation_data
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)

    # predict on the validation data
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    file_name = join(save_directory, 'bottleneck_features_validation.npy')
    np.save(open(file_name, 'w'), bottleneck_features_validation)

    return None


def softmax(nb_classes):
    """sets up a soft max with number of layers
    Args:
        nb_classes(int): number of classes
    Returns
        softmax(keras.models.Sequential)
    """

    # set up the
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def train_top_model(training_features_path, validation_feautures_path, top_model_weights_path, nb_classes,
                    train_labels, validation_labels, nb_epochs=50, batch_size=32):
    """ Trains a top softmax model with categorical crossentropy as the optimization function and save the weights.
        This is to initialize the weights
    Args:
        training_features_path(str): the path to the training features
        validation_feautures_path(str): the path to the validation features
        top_model_weights_path(str): path for saving the top model
        nb_classes(int): the number of classes for each
        train_samples_labels(list): a list of integers for the labels
        validation_samples_labels(list): a list of of integers for the validadtion labels
        nb_epochs(int): number of epochs to train
    Returns:
        the top model
    """

    # load the training and validatoin data
    train_data = np.load(open(training_features_path))
    validation_data = np.load(open(validation_feautures_path))

    # set up the softmax top model
    model = softmax(nb_classes=nb_classes)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose=0)

    model.save_weights(top_model_weights_path)
    return model
