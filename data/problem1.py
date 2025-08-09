from ..prelude import *
from ..data_abstract.spectrum_abstract import Spectrum
from . import DATA_RAW_PATH

__all__ = ["SpectrumProblem1"]


class SpectrumProblem1(Spectrum):
    def __init__(self) -> None:
        spd_data = []
        wavelength_data = []
        with open(DATA_RAW_PATH / "problem1.txt", encoding="utf8") as fin:
            next(fin)  # skip first row
            for ln in fin:
                wl, sp = map(float, ln.split())
                wavelength_data += [wl]
                spd_data += [sp * 1e-3]  # 单位转换
        self.spd_data: NDArray[np.float64] = np.array(spd_data)
        self.wavelength_data: NDArray[np.float64] = np.array(wavelength_data)

    def spd(self) -> NDArray[np.float64]:
        return self.spd_data

    def wavelength(self) -> NDArray[np.float64]:
        return self.wavelength_data
