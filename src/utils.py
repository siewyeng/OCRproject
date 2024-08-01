"""utils.py contains variables and constants to be used in other files"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

## code for type hinting
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

IS_GPU = False
PREFIX_LEN = 10  # increasing it may cause memory problems

D = torch.device
CPU = torch.device("cpu")
CUSTOM_PROMPT = ""

## for deployment
API_PORT = "8000"
API_HOST = "0.0.0.0"
