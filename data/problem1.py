from ..prelude import *
from ..data_abstract.spectrum_abstract import Spectrum
from . import DATA_RAW_PATH

__all__ = ["SpectrumProblem1"]


class SpectrumProblem1(Spectrum):
    def __init__(self) -> None:
        self.spd_data = []
        self.wavelength_data = []
        with open(DATA_RAW_PATH / "problem1.txt", encoding="utf8") as fin:
            next(fin)  # skip first row
            for ln in fin:
                wl, sp = map(float, ln.split())
                self.wavelength_data += [wl]
                self.spd_data += [sp * 1e-3] # 单位转换
        self.spd_data = np.array(self.spd_data)
        self.wavelength_data = np.array(self.wavelength_data)

    def spd(self) -> ArrayLike:
        return self.spd_data

    def wavelength(self) -> ArrayLike:
        return self.wavelength_data
