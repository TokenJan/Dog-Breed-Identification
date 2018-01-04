import importlib

def fRunCNN(dData, sModel, lTrain, lr, batchSize, epochs):
    # dynamic loading of corresponding model
    cnnModel = importlib.import_module(sModel, '.')
    # cnnModel = __import__(sModel, globals(), locals(), ['createModel', 'fTrain', 'fPredict'], -1)

    if lTrain:
        # train process
        cnnModel.fTrain(dData['x_trainFold'], dData['y_trainFold'], dData['x_valFold'], dData['y_valFold'], lr,
                        batchSize, epochs)
    else:
        # predict precess
        cnnModel.fPredict(dData['x_test'], batchSize)

