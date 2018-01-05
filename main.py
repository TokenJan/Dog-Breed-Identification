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

# load config params
sDatafile = cfg['sDatafile']

x_train = []
y_train = []
x_predict = []

if glob.glob(sDatafile):
    f = h5py.File(sDatafile, 'r')
    x_train = np.asarray(f['x_train'])
    y_train = np.asarray(f['y_train'])
    x_predict = np.asarray(f['x_predict'])

else:
    x_train, y_train, x_predict = datapre.fPreprocessData(cfg)

# print(x_train.shape)
# print(y_train.shape)
# print(x_predict.shape)

# number of the classes
nClass = y_train.shape[1]
#print(nClass)

if cfg['lCrossval']:
    # cross validation
    x_trainFold, y_trainFold, x_validFold, y_validFold  = datapre.crossValid(x_train, y_train, cfg['nFolds'])

    for iFold in range(cfg['nFolds']):
        # initialize training, validation and test data dictionary
        dData = {'x_train': x_trainFold[iFold], 'y_train': y_trainFold[iFold], 'x_valid': x_validFold[iFold],
                 'y_valid': y_validFold[iFold], 'x_predict': x_predict}

        # initialize parameter dictionary
        dParam = {'sModel': cfg['sModel'], 'lTrain': cfg['lTrain'], 'lr': cfg['lr'], 'batchSize': cfg['batchSize'],
                  'epochs': cfg['epochs'], 'img_size': cfg['img_size'], 'nClass': nClass}

        # start training or predicting
        cnn_main.fRunCNN(dData, dParam)

else:
    # split training data
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=cfg['validSplit'], random_state=1)

    # initialize training, validation and test data dictionary
    dData = {'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid, 'x_predict': x_predict}

    # initialize parameter dictionary
    dParam = {'sModel': cfg['sModel'], 'lTrain': cfg['lTrain'], 'lr': cfg['lr'], 'batchSize': cfg['batchSize'],
              'epochs': cfg['epochs'], 'img_size': cfg['img_size'], 'nClass': nClass}

    # start training or predicting
    cnn_main.fRunCNN(dData, dParam)