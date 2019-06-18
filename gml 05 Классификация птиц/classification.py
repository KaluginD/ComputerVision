import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing import image
import keras.layers as L
from keras.models import Model
from keras.applications.xception import preprocess_input, Xception

SHAPE = 300


def train_classifier(gt, img_dir, fast_train=True):
    images = []
    labels = []
    items = list(gt.items())[:200] if fast_train else gt.items()
    for name, label in items:
        img = image.load_img('{}/{}'.format(img_dir, name), target_size=(SHAPE, SHAPE))
        images.append(preprocess_input(image.img_to_array(img)))
        labels.append(label)
    labels = np.array(labels)
    images = np.array(images)
    classes = labels.max() + 1

    targets = LabelBinarizer().fit_transform(labels)
    X_train, X_test, Y_train, Y_test = train_test_split(images, targets, test_size=0.2)

    model_to_tune = Xception(weights='imagenet', include_top=False)
    pool = L.GlobalAveragePooling2D()(model_to_tune.output)
    dense = L.Dense(256, activation='relu')(pool)
    result = L.Dense(classes, activation='softmax')(dense)
    final_model = Model(inputs=model_to_tune.input, outputs=result)

    for layer in model_to_tune.layers:
        layer.trainable = False
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if fast_train:
        final_model.fit(X_train, Y_train, batch_size=128, epochs=1, validation_data=(X_test, Y_test))
        return final_model

    final_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))

    for layer in final_model.layers[126:]:
        layer.trainable = True
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    final_model.fit(X_train, Y_train, batch_size=128, epochs=25, validation_data=(X_test, Y_test))

    return final_model


def classify(model, img_dir):
    return {file:
            model.predict(
                   preprocess_input(
                       image.img_to_array(
                           image.load_img('/'.join([img_dir, file]), target_size=(SHAPE, SHAPE))
                       )
                   )[np.newaxis]
            )[0].argmax()
            for file in os.listdir(img_dir)}
