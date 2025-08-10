from .prelude import *
from .data.problem2 import Problem2Data
from .data_abstract.spectrum_abstract import Spectrum
from .parameters import calc_all_5_parameters

__all__ = [
    "p2s1_solution",
    "p2s2_solution",
    "CombinedSpectrum",
    "combine_spd",
]


def combine_spd(
    leds: list[Spectrum], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    按照配比组合光谱
    """
    result_spd = None
    for led, weight in zip(leds, weights):
        if result_spd is None:
            result_spd = led.spd() * weight
        else:
            result_spd += led.spd() * weight
    return cast(NDArray[np.float64], result_spd)


def optimize_scenario_1(problem2_data: Problem2Data) -> OptimizeResult:
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

    def to_minimize(weights: NDArray[np.float64]) -> np.float64:
        """
        要最小化的目标函数
        """
        stacked = np.vstack((anyone_wavelength, combine_spd(leds, weights)))
        return -cast(Any, spd_to_iesrf(stacked, "Rf"))[0][0]

    def constraints_fn(
        weights: NDArray[np.float64],
    ) -> tuple[np.float64, np.float64, np.float64]:
        """
        计算权值和、CCT、Rg
        """
        stacked = np.vstack((anyone_wavelength, combine_spd(leds, weights)))
        CCT, Rg = cast(Any, spd_to_iesrf(stacked, "cct,Rg"))
        return (
            np.sum(weights),
            CCT[0][0],
            Rg[0][0],
        )

    return differential_evolution(
        to_minimize,
        [(0, 1)] * 5,
        constraints=NonlinearConstraint(
            constraints_fn, np.array([1, 5500, 95]), np.array([1, 6500, 105])
        ),
        tol=1e-4,
    )


def optimize_scenario_2(problem2_data: Problem2Data) -> OptimizeResult:
    """
    构建场景二的解

    返回`OptimizeResult`

    目标函数
        min mel-DER
    约束条件
        2500<=CCT<=3500
        R_f>80
    """
    leds = problem2_data.five_leds()
    anyone_wavelength = leds[0].wavelength()

    def to_minimize(weights: NDArray[np.float64]) -> np.float64:
        """
        要最小化的目标函数
        """
        stacked = np.vstack((anyone_wavelength, combine_spd(leds, weights)))
        melDER = spd_to_aopicDER(stacked)[:, -1]
        return melDER[0]

    def constraints_fn(
        weights: NDArray[np.float64],
    ) -> tuple[np.float64, np.float64, np.float64]:
        """
        计算权值和、CCT、Rf
        """
        stacked = np.vstack((anyone_wavelength, combine_spd(leds, weights)))
        Rf, CCT = cast(Any, spd_to_iesrf(stacked, "Rf,cct"))
        return (
            np.sum(weights),
            CCT[0][0],
            Rf[0][0],
        )

    return differential_evolution(
        to_minimize,
        [(0, 1)] * 5,
        constraints=NonlinearConstraint(
            constraints_fn,
            np.array([1, 2500, 80]),
            np.array([1, 3500, np.inf]),
        ),
        tol=1e-4,
    )


class CombinedSpectrum(Spectrum):
    def __init__(self, leds: list[Spectrum], weights: NDArray[np.float64]) -> None:
        self.spd_data = combine_spd(leds, weights)
        self.wavelength_data = leds[0].wavelength()

    def spd(self):
        return self.spd_data

    def wavelength(self):
        return self.wavelength_data


class Problem2Solution(NamedTuple):
    weights: NDArray[np.float64]
    combined: CombinedSpectrum
    CCT: np.float64
    Duv: np.float64
    Rg: np.float64
    Rf: np.float64
    melDER: np.float64

    @classmethod
    def _calc(cls, data: Problem2Data, optimizer: Callable, identifier: str) -> Self:
        opt_res = optimizer(data)
        while not opt_res.success:
            warn(f"{identifier}遇到失败的解，正在重试……")
            opt_res = optimizer(data)

        weights = opt_res.x
        combined = CombinedSpectrum(data.five_leds(), weights)
        CCT, Duv, Rg, Rf, melDER = calc_all_5_parameters(combined)
        return cls(
            weights=weights,
            combined=combined,
            CCT=CCT,
            Duv=Duv,
            Rg=Rg,
            Rf=Rf,
            melDER=melDER,
        )


def p2s1_solution(data: Problem2Data) -> Problem2Solution:
    """
    问题二场景一的答案
    """
    return Problem2Solution._calc(data, optimize_scenario_1, "场景一")


def p2s2_solution(data: Problem2Data) -> Problem2Solution:
    """
    问题二场景二的答案
    """
    return Problem2Solution._calc(data, optimize_scenario_2, "场景二")
