from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import keras
import glob

# def addNewLayer(base_model, nClass):
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.8)(x)
#     x = Dense(1024, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.8)(x)
#
#     predictions = Dense(nClass, activation='softmax')(x)
#
#     model = Model(inputs=base_model.input, outputs=predictions)
#
#     return model

# def fineTune(model, opti):
#     for layer in model.layers[:-2]:
#         layer.trainable = False
#
#     for layer in model.layers[-2:]:
#         layer.trainable = True
#
#     if opti == 'sgd':
#         model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])
#
#     elif opti == 'adam':
#         optimizor = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
#         model.compile(loss='categorical_crossentropy',
#                       optimizer=optimizor,
#                       metrics=['accuracy'])
#
#     return model

def creatModel(img_size, nClass):
    # Create the base pre-trained model
    base_model = InceptionResNetV2(weights='imagenet',
                                   include_top=False,
                                   input_shape=(img_size[0], img_size[1], 3))

    # add last new layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    predictions = Dense(nClass, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model


def fTrain(dData, dParam, nClass):
    model_name = './model/' + dParam['sModel'] + '_' + str(dParam['img_size'][0]) + str(dParam['img_size'][1])\
                 + '_lr_' + str(dParam['lr']) + '_bs_' + str(dParam['batchSize'])

    model_all = model_name + '_model.h5'

    # load the model if it exists
    if glob.glob(model_all):
        model = load_model(model_all)
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

    model.summary()

    model.fit(dData['x_train'],
              dData['y_train'],
              epochs=dParam['epochs'],
              batch_size=dParam['batchSize'],
              validation_split=dParam['validSplit'],
              verbose=1)

    loss_test, acc_test = model.evaluate(dData['x_valid'], dData['y_valid'], batch_size=dParam['batchSize'], verbose=1)

    print("test loss: " + loss_test)
    print("test accuracy: " + acc_test)

    # save model
    model.save(model_all, overwrite=True)  # keras > v0.7

def fPredict(x_predict, batchSize):
    pass