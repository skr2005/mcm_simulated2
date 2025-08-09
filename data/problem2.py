from ..prelude import *
from ..data_abstract.spectrum_abstract import Spectrum
from . import DATA_RAW_PATH


__all__ = ["Problem2Data"]

class SpectrumProblem2(Spectrum):
    def __init__(self,led_num,arr1,arr2) ->None:
        self.n = led_num
        self.spd_data = arr1
        self.wavelength_data = arr2
    def spd(self)->NDArray[np.float64]:
        return self.spd_data
    def wavelength(self) -> NDArray[np.float64]:
        return self.wavelength_data

class Problem2Data:
    def __init__(self) -> None:
        data = pd.read_csv(DATA_RAW_PATH / "problem2.txt",sep="\s+",header=0,encoding="utf-8")
        wave_length = data.iloc[:,0].to_numpy(dtype=np.float64)
        leds = []
        for i in range(len(data.columns)-1): # 不算第一列波长列，所以减一
            leds.append(SpectrumProblem2(i,data.iloc[:,i+1].to_numpy(dtype=np.float64),wave_length)) # 对应的这里的spd列要加1
        self.leds = leds
    def five_leds(self) -> list[Spectrum]:
        """
        返回五种可用LED的光谱

        要求每次调用该函数返回的结果是相同的。
        要求五种LED的`Spectrum`的`wavelength`数组相同，可以让它们指向同一个数组。

        期望时间复杂度O(1)
        """
        return self.leds




