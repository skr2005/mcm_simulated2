from .prelude import *
from .data.problem2 import Problem2Data
from .data_abstract.spectrum_abstract import Spectrum


def scenario_1(problem2_data: Problem2Data) -> OptimizeResult:
    """
    构建场景一的解

    返回`OptimizeResult`

    目标函数
        max Rf
    约束条件
        5500 <= CCT <= 6500
        95 <= Rg <= 105
        88 < Rf （没有必要，因为已经是目标函数了）
    """
    leds = problem2_data.five_leds()
    anyone_wavelength = leds[0].wavelength()

    def combine(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        组合光谱
        """
        result_spd = None
        for led, weight in zip(leds, weights):
            if result_spd is None:
                result_spd = led.spd() * weight
            else:
                result_spd += led.spd() * weight
        return cast(NDArray[np.float64], result_spd)

    def to_minimize(weights: NDArray[np.float64]) -> np.float64:
        """
        要最小化的目标函数
        """
        stacked = np.vstack((anyone_wavelength, combine(weights)))
        return -cast(Any, spd_to_iesrf(stacked, "Rf"))[0][0]

    def constraints_fn(
        weights: NDArray[np.float64],
    ) -> tuple[np.float64, np.float64, np.float64]:
        """
        计算CCT、Rg、权值和
        """
        stacked = np.vstack((anyone_wavelength, combine(weights)))
        CCT, Rg = cast(Any, spd_to_iesrf(stacked, "CCT,Rg"))
        return CCT[0][0], Rg[0][0], np.sum(weights)

    return differential_evolution(
        to_minimize,
        [(0, 1)] * 5,
        workers=-1,
        constraints=NonlinearConstraint(
            constraints_fn, np.array([5500, 95, 1]), np.array([6500, 105, 1])
        ),
    )
