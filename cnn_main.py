from networks import top_layer
from keras.models import load_model
import pandas as pd
import glob

def fRunCNN(dData, nClass, cfg, sModel):
    # training process
    top_layer.fTrain(dData, nClass, cfg, sModel)

def fPredict(x_predict, dParam, sModel):
    model_file = './model/' + sModel + '_' + str(dParam['img_size'][0]) + '_bs_'\
                 + str(dParam['batchSize']) + '_model.h5'


    if glob.glob(model_file):
        model = load_model(model_file)
        y_predict = model.predict(x_predict, batch_size=dParam['batchSize'], verbose=1)

        df = pd.read_csv('./input/sample_submission.csv')

        for i, c in enumerate(df.columns[1:]):
            df[c] = y_predict[:, i]

        # save to file
        df.to_csv('./submission/{}.csv'.format(sModel), index=None)

    else:
        print('no such model')
