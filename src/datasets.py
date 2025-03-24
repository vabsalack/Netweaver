from netweaver.utils import cv2, os, np


def load_mnist_dataset(dataset, path):
    lables = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    for lable in lables:
        for file in os.listdir(os.path.join(path, dataset, lable)):
            image = cv2.imread(
                os.path.join(path, dataset, lable, file), cv2.IMREAD_UNCHANGED
            )
            X.append(image)
            y.append(lable)
    return np.array(X), np.array(y, dtype=np.int)

def create_data_mnist(path):
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)
    return X, y, X_test, y_test
