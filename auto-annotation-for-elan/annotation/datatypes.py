from typing import Tuple, Dict, Union, List

import numpy as np

NumpyArray = np.ndarray
TierAnnotation = Tuple[list, list, list]
Annotation = Dict[NumpyArray]
SceneList = Union[List, NumpyArray]
LocalMinima = List[Tuple[int, int], ...]