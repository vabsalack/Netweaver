from typing import List, Tuple, Union

import os
import cv2

import numpy as np
from numpy.typing import ArrayLike

import pickle
import copy

from netweaver.layers import LayerDense, LayerInput
from netweaver.activation_layers import ActivationSoftmax
from netweaver.lossfunctions import LossCategoricalCrossentropy
from netweaver.softmaxCCEloss import ActivationSoftmaxLossCategoricalCrossentropy


Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]
