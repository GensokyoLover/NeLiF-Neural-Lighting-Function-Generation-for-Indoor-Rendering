from typing import Sequence, Union, SupportsIndex, Optional
import random as rnd
import numpy as np

def random(shape: Optional[Union[SupportsIndex, Sequence[SupportsIndex]]] = None, dtype: None = np.float32):
    # [0.0, 1.0)
    return rnd.random() if shape is None else np.array([rnd.random() for _ in range(np.prod(shape))], dtype).reshape(shape)

def randint(a: int, b: int, shape: Optional[Union[SupportsIndex, Sequence[SupportsIndex]]] = None):
    # [a, b] (b is inclusive)
    return rnd.randint(a, b) if shape is None else np.array([rnd.randint(a, b) for _ in range(np.prod(shape))]).reshape(shape)
