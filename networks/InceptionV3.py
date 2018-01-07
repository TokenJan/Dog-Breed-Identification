from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import glob


def createModel(img_size, nClass):
    # Create the base pre-trained model

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Add a new top layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nClass, activation='softmax')(x)

    # This is the model to be trained
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model

def fTrain(dData, dParam, nClass):
    model_name = './model/' + dParam['sModel'] + '_' + str(dParam['img_size'][0]) + str(dParam['img_size'][1]) + '_lr_'\
                 + str(dParam['lr']) + '_bs_' + str(dParam['batchSize'])

    model_all = model_name + '_model.h5'

    # load the model if it exists
    if glob.glob(model_all):
        model = load_model(model_all)
    else:
        # initialize the model
        base_model, model = createModel(dParam['img_size'], nClass)

        # First: train only the top layers (which were randomly initialized)
        for layer in base_model.layers:
            layer.trainable = False

        if dParam['sOpti'] == 'rmsprop':
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        elif dParam['sOpti'] == 'adam':
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        elif dParam['sOpti'] == 'sgd':
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    callback_list = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    callback_list.append(ModelCheckpoint(model_all))
    callback_list.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    model.summary()

    model.fit(dData['x_train'],
              dData['y_train'],
              epochs=dParam['epochs'],
              batch_size=dParam['batchSize'],
              validation_split=dParam['validSplit'],
              verbose=1,
              callbacks=callback_list)

    loss_test, acc_test = model.evaluate(dData['x_valid'], dData['y_valid'], batch_size=dParam['batchSize'], verbose=1)

    print("test loss: " + loss_test)
    print("test accuracy: " + acc_test)

    # save model
    model.save(model_all, overwrite=True)   # keras > v0.7


def fPredict(x_predict, batchSize):
    pass