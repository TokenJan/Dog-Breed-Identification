from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import os

def creatModel(dData, nClass):
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=dData['train_data'].shape[1:]))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='sigmoid'))

    return model

def fTrain(dData, nClass, dParam, sModel):
    model_file = './model/' + sModel + '_' + str(dParam['img_size'][0]) + '_bs_' \
                 + str(dParam['batchSize']) + '_model.h5'

    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = creatModel(dData, nClass)

    model.compile(optimizer=dParam['sOpti'], loss='categorical_crossentropy', metrics=['accuracy'])

    callback_list = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001))
    callback_list.append(ModelCheckpoint(model_file))

    model.fit(dData['train_data'], dData['train_labels'],
                        epochs=dParam['epochs'],
                        batch_size=dParam['batchSize'],
                        validation_split=dParam['validSplit'],
                        callbacks=callback_list)

    metrics = model.evaluate(dData['valid_data'], dData['valid_labels'], batch_size=dParam['batchSize'], verbose=1)

    print('training data results: ')
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

