import numpy as np
import sys
import os
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

def run(cfg, generator_train, generator_test):
    lModel = cfg['lModel']

    if 'InceptionV3' in lModel:
        # build the InceptionV3 network
        model_train = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        file_train = './feature/InceptionV3_train.npy'
        model_train.summary()
        bottleneck_features_train = model_train.predict_generator(generator=generator_train, verbose=1)
        np.save(open(file_train, 'w'), bottleneck_features_train)
        print('InceptionV3 train bottleneck feature: {}'.format(bottleneck_features_train.shape))

        model_test = InceptionV3(include_top=False, weights='imagenet')
        file_test = './feature/InceptionV3_test.npy'
        bottleneck_features_test = model_test.predict_generator(generator=generator_test, verbose=1)
        np.save(open(file_test, 'w'), bottleneck_features_test)
        print('InceptionV3 test bottleneck feature: {}'.format(bottleneck_features_test.shape))

    if 'InceptionResNetV2' in lModel:
        # build the InceptionResNetV2 network
        model_train = InceptionResNetV2(include_top=False, weights='imagenet')
        file_train = './feature/InceptionResNetV2_train.npy'
        model_train.summary()
        bottleneck_features_train = model_train.predict_generator(generator=generator_train, verbose=1)
        np.save(open(file_train, 'w'), bottleneck_features_train)
        print('InceptionResNetV2 train bottleneck feature: {}'.format(bottleneck_features_train.shape))

        model_test = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
        file_test = './feature/InceptionResNetV2_test.npy'
        bottleneck_features_test = model_test.predict_generator(generator=generator_test, verbose=1)
        np.save(open(file_test, 'w'), bottleneck_features_test)
        print('InceptionResNetV2 test bottleneck feature: {}'.format(bottleneck_features_test.shape))

    else:
        sys.exit('Model is not supported.')




