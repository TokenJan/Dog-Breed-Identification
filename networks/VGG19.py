from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Dense
from keras.models import Model, load_model
import keras
import glob

def creatModel(img_size, nClass):
    # Create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Add a new top layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(nClass, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model

def fTrain(dData, dParam):
    model_name = './model/' + dParam['sModel'] + '_' + str(dParam['img_size'][0]) + str(dParam['img_size'][1]) + '_lr_'\
                 + str(dParam['lr']) + '_bs_' + str(dParam['batchSize'])

    model_all = model_name + '_model.h5'

    # load the model if it exists
    if glob.glob(model_all):
        model = load_model(model_all)
    else:
        # initialize the model
        base_model, model = creatModel(dParam['img_size'], dParam['nClass'])

        # First: train only the top layers (which were randomly initialized)
        for layer in base_model.layers:
            layer.trainable = False

        opti = keras.optimizers.Adam(lr=dParam['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opti,
                      metrics=['accuracy'])

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

    model.summary()

    model.fit(dData['x_train'],
              dData['y_train'],
              epochs=dParam['epochs'],
              batch_size=dParam['batchSize'],
              validation_split=0.1,
              verbose=1,
              callbacks=callbacks_list)

    model.evaluate(dData['x_valid'], dData['y_valid'], batch_size=dParam['batchSize'], verbose=1)

    # save model
    model.save(model_all, overwrite=True)   # keras > v0.7


def fPredict(x_predict, batchSize):
    pass