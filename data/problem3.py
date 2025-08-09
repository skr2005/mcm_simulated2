from ..data_abstract.spectrum_abstract import Spectrum
from ..prelude import *
from . import DATA_RAW_PATH

__all__ = ["Problem3Data"]

class SpectrumProblem3(Spectrum):
    def __init__(self, arr1, arr2) -> None:
        self.spd_data = arr1
        self.wavelength_data = arr2

    def spd(self) -> NDArray[np.float64]:
        return self.spd_data

    def wavelength(self) -> NDArray[np.float64]:
        return self.wavelength_data


class Problem3Data:
    def __init__(self) -> None:
        data = pd.read_csv(
            DATA_RAW_PATH / "problem3.txt", sep="\\s+", header=0, encoding="utf-8"
        )
        wave_length = data.iloc[:, 0].to_numpy(dtype=np.float64)
        ls = []
        for i in range(len(data.columns) - 1):  # 不算第一列波长列，所以减一
            ls.append(
                SpectrumProblem3(
                    data.iloc[:, i + 1].to_numpy(dtype=np.float64) * 1e-3, wave_length
                )
            )  # 对应的这里的spd列要加1
        cols = data.columns.tolist()
        self.time = cols[1:]
        self.time_table = ls

    def spectra(self) -> list[Spectrum]:
        """
        返回按照时间顺序的光谱列表

        预期时间复杂度：O(1)
        """
        return self.time_table
    
    def time_info(self) -> list[str]:
        """
        返回时间信息列表
        """
        return self.time
