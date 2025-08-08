from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from luxpy.toolboxes.photbiochem import spd_to_aopicDER
from luxpy.spectrum import spd
from luxpy.color.cri import spd_to_iesrf