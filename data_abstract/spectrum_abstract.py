from ..prelude import *


class Spectrum(ABC):
    @abstractmethod
    def spd(self) -> NDArray[np.float64]:
        """
        光强（功率分布数据），单位为W/m^2/nm

        例如它对应的是Problem 1的第二列乘上1e-3

        需要返回一个一维numpy数组，
        该数组每个位置的数据要和wavelength返回的每个位置的数据相对应

        期望时间复杂度：O(1)
        """

    @abstractmethod
    def wavelength(self) -> NDArray[np.float64]:
        """
        波长数据，单位为nm

        例如它对应的是Problem 1的第一列

        需要返回一个一维numpy数组，
        该数组每个位置的数据要和spd返回的每个位置的数据相对应

        期望时间复杂度：O(1)
        """
