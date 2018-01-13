import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50

def run(cfg):
    datagen = ImageDataGenerator(rescale=1. / 255)
    extractVGG19(datagen, cfg['batchSize'], cfg['img_size'])
    extractInceptionResNetV2(datagen, cfg['batchSize'], cfg['img_size'])
    extractResNet50(datagen, cfg['batchSize'], cfg['img_size'])

def extractVGG19(datagen, batch_size, img_size):
    # build the VGG19 network
    model = VGG19(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        directory='./input/train',
        target_size=(img_size[0], img_size[1]),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features = model.predict_generator(generator=generator, verbose=1)

    np.save(open('./featureVGG19_feature.npy', 'w'), bottleneck_features)

def extractInceptionResNetV2(datagen, batch_size, img_size):
    # build the InceptionResNetV2 network
    model = InceptionResNetV2(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        directory='./input/train',
        target_size=(img_size[0], img_size[1]),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features = model.predict_generator(generator=generator, verbose=1)

    np.save(open('./feature/InceptionResNetV2_feature.npy', 'w'), bottleneck_features)

def extractResNet50(datagen, batch_size, img_size):
    # build the ResNet50 network
    model = ResNet50(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        directory='./input/train',
        target_size=(img_size[0], img_size[1]),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features = model.predict_generator(generator=generator, verbose=1)

    np.save(open('./feature/ResNet50_feature.npy', 'w'), bottleneck_features)