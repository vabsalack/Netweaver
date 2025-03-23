from typing import List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from netweaver.layers import LayerDense, LayerInput
from netweaver.activation_layers import ActivationSoftmax
from netweaver.lossfunctions import LossCategoricalCrossentropy
from netweaver.fastbpsoftmaxcatcross import ActivationSoftmaxLossCategoricalCrossentropy


Float64Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]
