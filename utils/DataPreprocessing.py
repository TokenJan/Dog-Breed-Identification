import numpy as np
import pandas as pd
import cv2
import h5py
from sklearn.model_selection import KFold

# transform jpg pictures and corresponding labels into h5 file
def fPreprocessData(cfg):
    # load config param
    img_size = (cfg['img_size'][0], cfg['img_size'][1])

    # load csv data
    df_train = pd.read_csv('./input/labels.csv')
    df_predict = pd.read_csv('./input/sample_submission.csv')

    targets_series = pd.Series(df_train['breed'])
    one_hot = pd.get_dummies(targets_series, sparse = True)
    one_hot_labels = np.asarray(one_hot)

    # initialize the training and testing variable
    x_train = []
    y_train = []
    x_predict = []

    for i, row in df_train.iterrows():
        img = cv2.imread('./input/train/{}.jpg'.format(row.id))
        label = one_hot_labels[i]
        x_train.append(cv2.resize(img, img_size))
        y_train.append(label)
        
    for id in df_predict['id'].values:
        img = cv2.imread('./input/test/{}.jpg'.format(id))
        x_predict.append(cv2.resize(img, img_size))

    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.float32) / 255.
    x_predict  = np.array(x_predict, np.float32) / 255.

    with h5py.File('./dataset/dataset.h5', 'w') as hf:
        hf.create_dataset("x_train", data=x_train)
        hf.create_dataset("y_train", data=y_train)
        hf.create_dataset("x_predict", data=x_predict)

    return x_train, y_train, x_predict

def crossVal(x_data, y_data, nFolds):
    x_trainFold = []
    x_valFold = []
    y_trainFold = []
    y_valFold = []

    kf = KFold(n_splits=nFolds)
    
    for train_index, val_index in kf.split(x_data):
        x_train, x_val = x_data[train_index], x_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        x_trainFold.append(x_train)
        x_valFold.append(x_val)
        y_trainFold.append(y_train)
        y_valFold.append(y_val)

    return x_trainFold, y_trainFold, x_valFold, y_valFold   
