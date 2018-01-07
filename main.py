# external import
import yaml
import h5py
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# internal import
import utils.DataPreprocessing as datapre
import cnn_main

# load param configuration
with open('config/param.yml', 'r') as yml_file:
    cfg = yaml.safe_load(yml_file)

x_train = []
y_train = []
x_predict = []

df_train, df_predict, one_hot = datapre.fGetOnehot()

if glob.glob('./dataset/dataset' + '_' + str(cfg['img_size'][0]) + '.h5'):
    f = h5py.File('./dataset/dataset' + '_' + str(cfg['img_size'][0]) + '.h5', 'r')
    x_train = np.asarray(f['x_train'])
    y_train = np.asarray(f['y_train'])
    x_predict = np.asarray(f['x_predict'])

else:

    x_train, y_train, x_predict, one_hot = datapre.fPreprocessData(cfg, df_train, df_predict, one_hot)

# print(x_train.shape)
# print(y_train.shape)
# print(x_predict.shape)

# number of the classes
nClass = y_train.shape[1]
#print(nClass)

if cfg['lTrain']:
    if cfg['lCrossval']:
        # cross validation
        x_trainFold, y_trainFold, x_validFold, y_validFold  = datapre.crossValid(x_train, y_train, cfg['nFolds'])

        for iFold in range(cfg['nFolds']):
            # initialize training, validation and test data dictionary
            dData = {'x_train': x_trainFold[iFold], 'y_train': y_trainFold[iFold], 'x_valid': x_validFold[iFold],
                     'y_valid': y_validFold[iFold], 'x_predict': x_predict}

            # start training or predicting
            cnn_main.fRunCNN(dData, cfg, nClass)

    else:
        # split training data
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=cfg['validSplit'], random_state=1)

        # initialize training, validation and test data dictionary
        dData = {'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid, 'x_predict': x_predict}


        # start training or predicting
        cnn_main.fRunCNN(dData, cfg, nClass)
else:
    cnn_main.fPredict(x_predict, cfg, one_hot, df_predict)