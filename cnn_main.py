import importlib

def fRunCNN(dData, dParam, nClass):
    # dynamic loading of corresponding model
    cnnModel = importlib.import_module('networks.' + dParam['sModel'], '.')
    # cnnModel = __import__(sModel, globals(), locals(), ['createModel', 'fTrain', 'fPredict'], -1)

    if dParam['lTrain']:
        # train process
        cnnModel.fTrain(dData, dParam, nClass)
    else:
        # predict precess
        cnnModel.fPredict(dData['x_predict'], dParam['batchSize'])

