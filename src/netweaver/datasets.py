import os
import urllib
import urllib.request
from zipfile import ZipFile

import cv2
import numpy as np


def load_mnist_dataset(dataset, path) -> tuple[np.ndarray, np.ndarray]:
    """Loads the MNIST dataset.

    This function loads the MNIST dataset from the specified path.
    It reads images from subdirectories representing labels and returns them as numpy arrays.
    """
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(
                os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED
            )  # pass (cv2.IMREAD_UNCHANGED or -1) else it duplicates the same value for all channels, image shape (28, 28, 3)
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y).astype("uint8")


def object_size(ob):
    """Calculates the size of an object in gigabytes.

    This function uses the 'pympler' library to determine the size of the provided object
    and returns the result formatted as a string with units in gigabytes.

    Parameters
    ----------
    ob : object
        The object whose size needs to be calculated.

    Returns
    -------
    str
        The size of the object in gigabytes, formatted as a string.
    """
    from pympler import asizeof

    object_bytes = asizeof.asizeof(ob)
    return f"{object_bytes / 1073741824:.3f}Gb used"


def create_data_mnist(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates training and testing data for the MNIST dataset.

    This function loads the MNIST dataset from the specified directory, which must contain
    'train' and 'test' subdirectories. It reads the images and their corresponding labels
    from these subdirectories and returns them as numpy arrays.

    Parameters
    ----------
    path : str
        The path to the directory containing 'train' and 'test' subdirectories.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing four numpy arrays:
        - Training data (X)
        - Training labels (y)
        - Testing data (X_test)
        - Testing labels (y_test)
    """
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)
    print(
        f"X shape:      {X.shape} dtype({type(X[0][0][0])}) mem usage: {object_size(X)}, y shape:      {y.shape} dtype({type(y[0])}) mem usage: {object_size(y)}"
    )
    print(
        f"X_test shape: {X_test.shape} dtype({type(X_test[0][0][0])}) mem usage: {object_size(X_test)}, y_test shape: {y_test.shape} dtype({type(y_test[0])}) mem usage: {object_size(y_test)}"
    )
    return X, y, X_test, y_test


def extract_file(file: str, folder: str) -> None:
    print("Unzipping images...")
    with ZipFile(file) as zip_images:
        zip_images.extractall(folder)


def download_file(url: str, file_name: str) -> None:
    print(f"Downloading {url} and saving as {file_name}...")
    urllib.request.urlretrieve(url, file_name)


def download_fashion_mnist_dataset(project_root_path) -> None:
    """
    Downloads and sets up the Fashion MNIST dataset.

    This function creates a 'datasets' directory in the root project folder if it doesn't exist.
    It then checks for a 'fashion_mnist_images' folder within the datasets directory.
    If the folder is not present, it downloads the Fashion MNIST dataset and extracts it.

    Parameters
    ----------
    project_root_path : str
        The path to the root directory of the project.
    returns
    -------
    None
        This function does not return anything. It creates a directory and downloads files.
    """
    FOLDER = "fashion_mnist_images"

    # cwd_path = os.getcwd()
    # dataset_pdir_path = (
    #     os.path.dirname(os.path.dirname(cwd_path))
    #     if "src/netweaver" in cwd_path
    #     else cwd_path
    # ) idea: find the project root by .venv folder. using os.exectuables, prfix or path
    datasetdir_path = f"{project_root_path}/datasets"
    # checks for "datasets" directory in project directory, if not. It creates one.
    if not os.path.isdir(datasetdir_path):
        os.mkdir(datasetdir_path)

    if not os.path.isdir(f"{datasetdir_path}/{FOLDER}"):
        URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
        FILE = "fashion_mnist_images.zip"
        download_file(URL, f"{datasetdir_path}/{FILE}")

        extract_file(f"{datasetdir_path}/{FILE}", f"{datasetdir_path}/{FOLDER}")
        print("add datasets/ dir to .gitignore file if versioned")
    else:
        print(f"Fashion_mnish_dataset is already available in {datasetdir_path}/{FOLDER}")
