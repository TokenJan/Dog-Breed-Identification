from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
import glob

def createModel(img_size, nClass):
    # Create the base pre-trained model
    base_model = InceptionResNetV2(weights='imagenet',
                                   include_top=False,
                                   input_shape=(img_size[0], img_size[1], 3))

    # add last new layers
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.2)(x)

    predictions = Dense(nClass, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model

def fTrain(dData, dParam, nClass):
    for lr in dParam['lr']:
        fTrainInner(dData, dParam, nClass, lr)

def fTrainInner(dData, dParam, nClass, lr):
    model_file = './model/' + dParam['sModel'] + '_' + str(dParam['img_size'][0]) + '_bs_'\
                 + str(dParam['batchSize']) + '_model.h5'

    # load the model if it exists
    if glob.glob(model_file):
        model = load_model(model_file)
    else:
        # initialize the model
        base_model, model = createModel(dParam['img_size'], nClass)

        # First: train only the top layers (which were randomly initialized)
        for layer in base_model.layers:
            layer.trainable = False

        if dParam['sOpti'] == 'rmsprop':
            rmsprop = optimizers.rmsprop(lr=lr)
            model.compile(optimizer=rmsprop,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        elif dParam['sOpti'] == 'adam':
            adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=adam,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        elif dParam['sOpti'] == 'sgd':
            sgd = optimizers.SGD(lr=lr, momentum=0.9)
            model.compile(optimizer=sgd,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    callback_list = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    callback_list.append(ModelCheckpoint(model_file))
    callback_list.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    model.summary()

    model.fit(dData['x_train'],
              dData['y_train'],
              epochs=dParam['epochs'],
              batch_size=dParam['batchSize'],
              validation_split=dParam['validSplit'],
              verbose=1,
              callbacks=callback_list)

    metrics = model.evaluate(dData['x_valid'], dData['y_valid'], batch_size=dParam['batchSize'], verbose=1)

    print('training data results: ')
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

    # save model
    # model.save(model_file, overwrite=True)  # keras > v0.7