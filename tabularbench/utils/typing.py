"""This module containts type aliases for this project.
"""

from typing import Union

import numpy as np
import numpy.typing as npt

NDNumber = npt.NDArray[Union[np.float64, np.int64]]
NDFloat = npt.NDArray[np.float64]
NDInt = npt.NDArray[np.int64]
NDBool = npt.NDArray[np.bool_]
