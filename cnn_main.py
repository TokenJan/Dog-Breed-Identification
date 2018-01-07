import importlib
from keras.models import load_model
import pandas as pd
import glob

def fRunCNN(dData, dParam, nClass):
    # dynamic loading of corresponding model
    cnnModel = importlib.import_module('networks.' + dParam['sModel'], '.')

    # training process
    cnnModel.fTrain(dData, dParam, nClass)

def fPredict(x_predict, dParam, one_hot, df_predict):
    model_file = './model/' + dParam['sModel'] + '_' + str(dParam['img_size'][0]) + '_bs_'\
                 + str(dParam['batchSize']) + '_model.h5'


    if glob.glob(model_file):
        model = load_model(model_file)
        y_predict = model.predict(x_predict, batch_size=dParam['batchSize'], verbose=1)

        sub = pd.DataFrame(y_predict)

        # Set column names to those generated by the one-hot encoding earlier
        col_names = one_hot.columns.values
        sub.columns = col_names

        # Insert the column id from the sample_submission at the start of the data frame
        sub.insert(0, 'id', df_predict['id'])

        # save to file

        sub.to_csv('./submission/result.csv')

    else:
        print('no such model')
