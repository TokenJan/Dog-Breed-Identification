from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Dense
from keras.models import Model
import keras

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
    # initialize the model
    base_model, model = creatModel(dParam['img_size'], dParam['nClass'])

    # First: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()

    model.fit(dData['x_train'], dData['y_train'], epochs=1, validation_data=(dData['x_val'], dData['y_val']), verbose=1)

def fPredict(x_test, batchSize):
    pass