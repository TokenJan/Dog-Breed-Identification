import importlib
from keras.models import load_model
import pandas as pd
import glob

def fRunCNN(dData, dParam, nClass):
    # dynamic loading of corresponding model
    cnnModel = importlib.import_module('networks.' + dParam['sModel'], '.')

    # training process
    cnnModel.fTrain(dData, dParam, nClass)

def fPredict(x_predict, dParam):
    model_file = './model/' + dParam['sModel'] + '_' + str(dParam['img_size'][0]) + '_bs_'\
                 + str(dParam['batchSize']) + '_model.h5'


    if glob.glob(model_file):
        model = load_model(model_file)
        y_predict = model.predict(x_predict, batch_size=dParam['batchSize'], verbose=1)

        df = pd.read_csv('./input/sample_submission.csv')

        for i, c in enumerate(df.columns[1:]):
            df[c] = y_predict[:, i]

        # save to file
        df.to_csv('./submission/pred.csv', index=None)

    else:
        print('no such model')
