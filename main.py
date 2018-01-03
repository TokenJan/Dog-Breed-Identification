# external import
import yaml
import h5py
import glob
import numpy as np

# internal import
import utils.DataPreprocessing as datapre

# load param configuration
with open('config/param.yml', 'r') as yml_file:
    cfg = yaml.safe_load(yml_file)

# load config params
sDatafile = cfg['sDatafile']

x_train = []
y_train = []
x_test = []

if glob.glob(sDatafile):
    f = h5py.File(sDatafile, 'r')
    x_train = np.asarray(f['x_train'])
    y_train = np.asarray(f['y_train'])
    x_test = np.asarray(f['x_test'])

else:
    x_train, y_train, x_test = datapre.fPreprocessData(cfg)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)

# cross validation
x_trainFold, y_trainFold, x_valFold, y_valFold  = datapre.crossVal(x_train, y_train, cfg['nFolds'])