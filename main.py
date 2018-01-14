# external import
import yaml
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# internal import
import utils.DataPreprocessing as datapre
import utils.FeatureExtractor as feature_extractor
import cnn_main

# load param configuration
with open('config/param.yml', 'r') as yml_file:
    cfg = yaml.safe_load(yml_file)

# initialize the data generator
img_size = cfg['img_size']
batch_size = cfg['batchSize']
datagen = ImageDataGenerator(rescale=1. / 255)
generator_train = datagen.flow_from_directory(
    directory='./input/train',
    target_size=(img_size[0], img_size[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

generator_test = datagen.flow_from_directory(
    directory='./input/test',
    target_size=(img_size[0], img_size[1]),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

if glob.glob('./input/train/*.jpg'):
    datapre.sortImg(cfg)

if cfg['lExtractor']:
    feature_extractor.run(cfg, generator_train, generator_test)

elif cfg['lTrain']:
    nTrain = len(generator_train.filenames)
    nClass = len(generator_train.class_indices)

    if cfg['lAssemble']:
        train_data_InceptionV3 = np.load('./feature/InceptionV3_train.npy')
        train_data_InceptionResNetV2 = np.load('./feature/InceptionResNetV2_train.npy')
        train_data_original = np.concatenate((train_data_InceptionV3, train_data_InceptionResNetV2), axis=1)

        # get the class lebels for the training data, in the original order
        train_labels_original = generator_train.classes
        # convert the training labels to categorical vectors
        train_labels_original = to_categorical(train_labels_original, num_classes=nClass)

        # split training data
        for _ in range(cfg['nFolds']):
            train_data, valid_data, train_labels, valid_labels = train_test_split(train_data_original, train_labels_original,
                                                                                  test_size=cfg['validSplit'],
                                                                                  random_state=1)

            dData = {'train_data': train_data, 'valid_data': valid_data, 'train_labels': train_labels,
                     'valid_labels': valid_labels}

            cnn_main.fRunCNN(dData, nClass, cfg, 'assemble')


    else:
        # load the bottleneck features saved earlier
        for sModel in cfg['lModel']:
            train_data = np.load('./feature/{}_train.npy'.format(sModel))

            # get the class lebels for the training data, in the original order
            train_labels = generator_train.classes

            # convert the training labels to categorical vectors
            train_labels = to_categorical(train_labels, num_classes=nClass)

            # split training data
            for _ in range(cfg['nFolds']):
                train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels,
                                                                                      test_size=cfg['validSplit'],
                                                                                      random_state=1)

                dData = {'train_data': train_data, 'valid_data': valid_data, 'train_labels': train_labels,
                         'valid_labels': valid_labels}

                cnn_main.fRunCNN(dData, nClass, cfg, sModel)

else:
    if cfg['lAssemble']:
        predict_data_InceptionV3 = np.load('./feature/InceptionV3_train.npy')
        predict_data_InceptionResNetV2 = np.load('./feature/InceptionResNetV2_train.npy')
        predict_data = np.concatenate((predict_data_InceptionV3, predict_data_InceptionResNetV2), axis=1)
        cnn_main.fPredict(predict_data, cfg, 'assemble')
    else:
        for sModel in cfg['lModel']:
            predict_data = np.load('./feature/{}_test.npy'.format(sModel))
            cnn_main.fPredict(predict_data, cfg, sModel)