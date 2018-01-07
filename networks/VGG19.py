from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import glob

def creatModel(img_size, nClass):
    # Create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Add a new top layer
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.8)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.8)(x)

    # This is the model to be train
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model

def fTrain(dData, dParam, nClass):
    model_file = './model/' + dParam['sModel'] + '_' + str(dParam['img_size'][0]) + '_lr_'\
                 + str(dParam['lr']) + '_bs_' + str(dParam['batchSize']) + '_model.h5'

    # load the model if it exists
    if glob.glob(model_file):
        model = load_model(model_file)
    else:
        # initialize the model
        base_model, model = creatModel(dParam['img_size'], nClass)

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

    loss_test, acc_test = model.evaluate(dData['x_valid'], dData['y_valid'], batch_size=dParam['batchSize'], verbose=1)

    print("test loss: " + loss_test)
    print("test accuracy: " + acc_test)

    # save model
    # model.save(model_file, overwrite=True)   # keras > v0.7
