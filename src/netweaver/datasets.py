import os
import urllib
import urllib.request
from zipfile import ZipFile

import matplotlib.image as mpimg
import numpy as np
from pympler import asizeof

from ._internal_utils import _create_directory, get_cwd, join_path


def _load_data(path, dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads images and labels from a specified dataset directory.

    This function reads all images and their corresponding labels from the given dataset folder and returns them as numpy arrays.

    Parameters
    ----------
    path : str
        The base directory containing the dataset.
    dataset : str
        The name of the dataset subdirectory (e.g., 'train' or 'test').

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the image data as a float32 numpy array and the labels as a uint8 numpy array.
    """
    labels = os.listdir(os.path.join(path, dataset))
    instances = []
    gtruth = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = mpimg.imread(os.path.join(path, dataset, label, file))
            instances.append(image)
            gtruth.append(label)
    return np.array(instances, dtype=np.float32), np.array(gtruth, dtype=np.uint8)


def _object_size(python_object):
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

    object_bytes = asizeof.asizeof(python_object)
    return f"{object_bytes / 1073741824:.5f}"


def _summary_dataset(instances_train: np.ndarray, gtruth_train: np.ndarray, instances_test: np.ndarray, gtruth_test: np.ndarray) -> str:
    """Summarizes key properties of training and testing datasets.

    This function takes training and testing data and labels as input, then computes and
    formats a summary of their key properties, including instance count, shape of dataset, shape of instance, data type,
    and memory usage.

    Parameters
    ----------
    instances_train : np.ndarray
        Training data instances.
    gtruth_train : np.ndarray
        Training data labels.
    instances_test : np.ndarray
        Testing data instances.
    gtruth_test : np.ndarray
        Testing data labels.

    Returns
    -------
    str
        A formatted string containing the summary table of dataset properties.
    """
    # Collect all arrays into a dict for convenience
    datasets = {"instances_train": instances_train, "gtruth_train": gtruth_train, "instances_test": instances_test, "gtruth_test": gtruth_test}
    # Prepare headers
    headers = list(datasets.keys())
    column_width = max(len(h) for h in headers) + 2  # Create table rows

    rows = [
        ("instances count", [arr.shape[0] for arr in datasets.values()]),
        ("shape of the set", [arr.shape for arr in datasets.values()]),
        ("shape of an instance", [arr.shape[1:] if arr.ndim > 1 else (1,) for arr in datasets.values()]),
        ("Data type of unit", [str(arr.dtype) for arr in datasets.values()]),
        ("Total memory (gb)", [_object_size(arr) for arr in datasets.values()]),
    ]

    # Print header row
    header_row = "Property".ljust(25) + "| " + "".join(h.ljust(column_width) for h in headers)
    output_text = [header_row, "-" * len(header_row)]
    # Print each data row
    for label, values in rows:
        row = f"{label.ljust(25)}| "
        for val in values:
            row += str(val).ljust(column_width)
        output_text.append(row)

    return "\n".join(output_text)


def load_dataset(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates training and testing data for the dataset (path).

    This function loads the dataset from the specified directory, which must contain
    'train' and 'test' subdirectories. Each sample or instances should be placed in a folder named after its label indices (zero-based indexing).
    The function reads all images and their associated labels from these subdirectories, prints Summarization and returns them as numpy arrays.

    Parameters
    ----------
    path : str
        The path to the directory containing 'train' and 'test' subdirectories.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing four numpy arrays:
        - Training data (instances_train)
        - Training labels (gtruth_train)
        - Testing data (instances_test)
        - Testing labels (gtruth_test)
    """
    instances_train, gtruth_train = _load_data(path, "train")
    instances_test, gtruth_test = _load_data(path, "test")

    print(_summary_dataset(instances_train, gtruth_train, instances_test, gtruth_test))
    return instances_train, gtruth_train, instances_test, gtruth_test


def _extract_file(file: str, folder: str) -> None:
    """Extracts a zip file to a specified folder.

    This function extracts all the contents of a given zip file to the designated folder.

    Parameters
    ----------
    file : str
        The path to the zip file to extract.
    folder : str
        The path to the folder where the zip file contents should be extracted.

    Returns
    -------
    None
        This function does not return anything. It extracts the zip file.
    """
    print("Unzipping images...")
    with ZipFile(file) as zip_images:
        zip_images.extractall(folder)


def _download_file(url: str, file_name: str) -> None:
    """Downloads a file from a given URL and saves it to a specified location.

    This function retrieves a file from the provided URL using urllib.request.urlretrieve
    and saves it with the given file name.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    file_name : str
        The name to save the downloaded file as.

    Returns
    -------
    None
        This function does not return anything. It downloads and saves a file.
    """
    print(f"Downloading {url} and saving as {file_name}...")
    urllib.request.urlretrieve(url, file_name)


def download_fashion_mnist_dataset(path_project_root=get_cwd()) -> None:
    """
    Downloads and sets up the Fashion MNIST dataset.

    This function creates a 'datasets' directory in the root project folder if it doesn't exist.
    It then checks for a 'fashion_mnist_images' folder within the datasets directory.
    If the folder is not present, it downloads the Fashion MNIST dataset and extracts it.

    Parameters
    ----------
    path_project_root : str, Optional
        The path to the root directory of the project. by default current working directory.

    Returns
    -------
    None
        This function does not return anything. It creates a directory and downloads files.
    """
    folder = "fashion_mnist_images"

    datasetdir_path = join_path(path_project_root, "datasets")
    _create_directory(datasetdir_path)

    if not os.path.isdir(join_path(datasetdir_path, folder)):
        _download_extract_fashion_mnist(datasetdir_path, folder)
    else:
        print(f"Fashion_mnish_dataset is already available in {join_path(datasetdir_path, folder)}")


def _download_extract_fashion_mnist(datasetdir_path, folder):
    """
    Downloads and extracts the Fashion MNIST dataset zip file.

    This function downloads the Fashion MNIST zip file from a predefined URL, saves it to the specified dataset directory,
    and extracts its contents into the given folder.

    Parameters
    ----------
    datasetdir_path : str
        The path to the directory where the dataset zip file will be downloaded.
    folder : str
        The folder where the extracted files will be placed.

    Returns
    -------
    None
        This function does not return anything. It downloads and extracts the dataset.
    """
    url = "https://nnfs.io/datasets/fashion_mnist_images.zip"
    file = "fashion_mnist_images.zip"

    path_download_zipfile = join_path(datasetdir_path, file)
    _download_file(url, path_download_zipfile)

    path_extract_folder = join_path(datasetdir_path, folder)
    _extract_file(path_download_zipfile, path_extract_folder)

    print(f"Please add datasets/ dir to .gitignore file if versioned\nFashion MNIST cooked and available in: {path_extract_folder}")
