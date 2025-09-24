from typing import Optional, Union, List, Dict, Any
import xml.etree.ElementTree as ET

import numpy as np
from pympi.Elan import Eaf

FilePath = Optional[str]
FolderPath = Optional[str]
EafFile = Eaf
NumpyArray = np.ndarray
Timeseries = Dict[NumpyArray]
SceneList = Union[List, NumpyArray]
Number = Union[float, int]
Annotations = Any
XMLTree = ET.ElementTree
XMLRoot = ET.Element