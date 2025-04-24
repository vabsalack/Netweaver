from netweaver.datasets import create_data_mnist, load_mnist_dataset

import pytest

@pytest.fixture
def test_data_filepath():
    test_dict = {"dataset_abs_path": "/home/keerthivasan_user/Documents/git/netweaver/datasets"}
    return test_dict

def test_load_mnist_dataset():
    pass

def test_create_data_mnist():
    pass