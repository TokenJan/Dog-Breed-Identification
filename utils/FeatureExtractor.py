import numpy as np
import sys
import os
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50

def run(cfg, generator_train, generator_test):
    sModel = cfg['sModel']

    if sModel == 'VGG19':
        # build the VGG19 network
        if not os.path.exists('./feature/VGG19_train.npy'):
            model_train = VGG19(include_top=False, weights='imagenet')
            file_train = './feature/VGG19_train.npy'

        if not os.path.exists('./feature/VGG19_test.npy'):
            model_test = VGG19(include_top=False, weights='imagenet')
            file_test = './feature/VGG19_test.npy'
    elif sModel == 'InceptionResNetV2':
        # build the InceptionResNetV2 network
        if not os.path.exists('./feature/InceptionResNetV2_train.npy'):
            model_train = InceptionResNetV2(include_top=False, weights='imagenet')
            file_train = './feature/InceptionResNetV2_train.npy'

        if not os.path.exists('./feature/InceptionResNetV2_test.npy'):
            model_test = InceptionResNetV2(include_top=False, weights='imagenet')
            file_test = './feature/InceptionResNetV2_test.npy'
    elif sModel == 'ResNet50':
        # build the ResNet50 network
        if not os.path.exists('./feature/ResNet50_train.npy'):
            model_train = ResNet50(include_top=False, weights='imagenet')
            file_train = './feature/ResNet50_train.npy'

        if not os.path.exists('./feature/ResNet50_test.npy'):
            model_train = ResNet50(include_top=False, weights='imagenet')
            file_test = './feature/ResNet50_test.npy'
    else:
        sys.exit('Model is not supported.')

    bottleneck_features = model_train.predict_generator(generator=generator_train, verbose=1)
    np.save(open(file_train, 'w'), bottleneck_features)

    bottleneck_features = model_test.predict_generator(generator=generator_train, verbose=1)
    np.save(open(file_test, 'w'), bottleneck_features)
