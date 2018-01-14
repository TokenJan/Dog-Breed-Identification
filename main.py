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

    # load the bottleneck features saved earlier
    for sModel in cfg['lModel']:
        train_data = np.load('./feature/{}_train.npy'.format(sModel))

        # get the class lebels for the training data, in the original order
        train_labels = generator_train.classes

        # convert the training labels to categorical vectors
        train_labels = to_categorical(train_labels, num_classes=nClass)

        # split training data
        if cfg['lCrossval']:
            x_trainFold = []
            x_valFold = []
            y_trainFold = []
            y_valFold = []
            kf = KFold(n_splits=cfg['nFolds'])

            for train_index, val_index in kf.split(train_data):
                x_train, x_val = train_data[train_index], train_data[val_index]
                y_train, y_val = train_labels[train_index], train_labels[val_index]

                dData = {'train_data': x_train, 'valid_data': x_val, 'train_labels': y_train,
                         'valid_labels': y_val}

                cnn_main.fRunCNN(dData, nClass, cfg, sModel)

        else:
            train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=cfg['validSplit'], random_state=1)

            dData = {'train_data': train_data, 'valid_data': valid_data, 'train_labels': train_labels, 'valid_labels': valid_labels}

            cnn_main.fRunCNN(dData, nClass, cfg, sModel)

else:
    for sModel in cfg['lModel']:
        predict_data = np.load('./feature/{}_test.npy'.format(sModel))
        cnn_main.fPredict(predict_data, cfg, sModel)