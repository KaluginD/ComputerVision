import numpy as np
from itertools import chain
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.ndimage import filters
from sklearn.svm import LinearSVC


def extract_hog(color_image):
    gray_image = rgb2gray(color_image)
    image = resize(gray_image, (32, 32))
    i_x = filters.convolve(image, np.array([[-1, 0, 1]]), mode='constant')
    i_y = filters.convolve(image, np.array([[-1], [0], [1]]), mode='constant')
    grad_norm = np.sqrt(i_x ** 2 + i_y ** 2)
    direction = np.arctan2(i_y, i_x)
    direction[direction < 0] += np.pi
    hog = []
    for x in range(0, 16, 2):
        for y in range(0, 16, 2):
            vectors = [np.histogram(direction[x: x + 16, y: y + 16][i * 8: i * 8 + 8, j * 8: j * 8 + 8].flatten(),
                       8, range=(0, np.pi),
                       weights=grad_norm[x: x + 16, y: y + 16][i * 8: i * 8 + 8, j * 8: j * 8 + 8].flatten())[0]
                       for i in range(2) for j in range(2)]
            vectors = np.array(list(chain(*vectors)))
            vectors = vectors / np.sqrt(np.sum(vectors ** 2) + 1e-8)
            hog.extend(vectors)
    return np.array(hog)


def fit_and_classify(x_train, y_train, x_test):
    model = LinearSVC(C=0.25)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred
