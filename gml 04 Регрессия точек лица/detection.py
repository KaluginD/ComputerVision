import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
from keras import models
import keras.layers as L
from skimage.transform import resize
from skimage import io

INPUT_SHAPE = (100, 100, 3)


def train_detector(gt, img_dir, fast_train=True):
    points_cords, images = [], []
    items = list(gt.items())[:500] if fast_train else gt.items()
    for image_name, points in items:
        image = io.imread('{}/{}'.format(img_dir, image_name))
        images.append(resize(image, INPUT_SHAPE))
        points = np.array(points, dtype=float)
        points[0::2] *= 100. / image.shape[1]
        points[1::2] *= 100. / image.shape[0]
        points_cords.append(points)
    points_cords, images = np.array(points_cords), np.array(images)

    model = models.Sequential()
    for i in range(6):
        if i == 0:
            model.add(L.Convolution2D(filters=16, kernel_size=3, input_shape=INPUT_SHAPE))
        else:
            model.add(L.Convolution2D(filters=16 * 2**i, kernel_size=3))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
        if i % 2 == 1:
            model.add(L.MaxPooling2D())

    model.add(L.Flatten())
    for units in [64, 64, 2 * 14]:
        model.add(L.Dense(units=units, activation='relu'))

    X_train, X_test, Y_train, Y_test = train_test_split(images, points_cords, test_size=0.1)
    model.compile('adam', loss='mse')

    if fast_train:
        model.fit(X_train, Y_train, batch_size=256, epochs=1, validation_data=(X_test, Y_test))
        return model
    model.fit(X_train, Y_train, batch_size=512, epochs=100, validation_data=(X_test, Y_test))

    X_train, X_test, Y_train, Y_test = train_test_split(images, points_cords, test_size=0.1)
    model.compile('adam', loss='mse')
    model.fit(X_train, Y_train, batch_size=256, epochs=100, validation_data=(X_test, Y_test))
    return model


def detect(model, test_img_dir):
    results = {}
    for i, image_name in enumerate(listdir(test_img_dir)):
        # if i % 100 == 0:
        #    print(i)
        image = io.imread('{}/{}'.format(test_img_dir, image_name))
        points = model.predict(resize(image, INPUT_SHAPE)[np.newaxis])[0]
        # w, h _ = image.shape
        points[0::2] /= 100. / image.shape[1]
        points[1::2] /= 100. / image.shape[0]
        results[image_name] = points
    return results
