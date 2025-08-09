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
        计算权值和、CCT、Rg
        """
        stacked = np.vstack((anyone_wavelength, combine(weights)))
        CCT, Rg = cast(Any, spd_to_iesrf(stacked, "cct,Rg"))
        return np.sum(weights), CCT[0][0], Rg[0][0], 

    return differential_evolution(
        to_minimize,
        [(0, 1)] * 5,
        constraints=NonlinearConstraint(
            constraints_fn, np.array([1, 5500, 95]), np.array([1, 6500, 105])
        ),
    )

"""
scenario_1(Problem2Data())

res.x = array([1.5052e-01, 1.7068e-01, 2.4089e-01, 9.6369e-03, 4.2827e-01])
res.fun = -92.85323145954435
"""