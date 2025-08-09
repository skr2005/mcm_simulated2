from abc import ABC, abstractmethod
from typing import Any, cast
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import differential_evolution, NonlinearConstraint, OptimizeResult
from luxpy.toolboxes.photbiochem import spd_to_aopicDER
from luxpy.spectrum import spd
from luxpy.color.cri import spd_to_iesrf