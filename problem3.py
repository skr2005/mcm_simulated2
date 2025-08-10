from .prelude import *
from .data.problem3 import Problem3Data
from .parameters import calc_melDER
from .data_abstract.spectrum_abstract import Spectrum
from .problem2 import CombinedSpectrum


def p3_melDERs(data: Problem3Data) -> NDArray[np.float64]:
    return np.array(list(map(calc_melDER, data.spectra())))


def p3_plot_melDER(data3: Problem3Data):
    """
    画出目标melDER关于时间的散点图
    """
    y = []
    for spectrum in data3.spectra():
        stacked = np.vstack((spectrum.wavelength(), spectrum.spd()))
        melDER = spd_to_aopicDER(stacked)[:, -1]
        y.append(melDER[0])
    x = [i + 5.5 for i in range(len(data3.time_info()))]
    plt.scatter(x, y, s=4, color="blue")
    print(x)
    print(y)
    plt.legend()
    plt.xlabel("时间")
    plt.ylabel("mel-DER")
    plt.title("mel-DER随时间的变化趋势")
    plt.show()


def fit_melDER(
    leds: list[Spectrum],
    target_melDER: np.float64,
    initial_guess: ArrayLike | None = None,
) -> OptimizeResult:
    """
    通过组合LED拟合目标mel-DER，返回各个LED的权重。
    权重非负，但和不一定为1，需要手动正规化。
    """
    if initial_guess is None:
        initial_guess = [1 / len(leds)] * len(leds)
    return least_squares(
        lambda w: calc_melDER(CombinedSpectrum(leds, w / np.sum(w))) - target_melDER,
        initial_guess,
        bounds=(0, 1),
    )


def solve_problem3_weights(
    leds: list[Spectrum], target_melDERs: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    拟合第三问的各个目标光谱的melDER，
    返回各个权重（非负且和为1）以及残差
    """
    w_matrix = []
    residual = []
    for melDER in target_melDERs:
        fit_res = fit_melDER(leds, melDER, w_matrix[-1] if w_matrix else None)
        if not fit_res.success:
            raise fit_res.message
        w_matrix += [fit_res.x / np.sum(fit_res.x)]
        residual += [fit_res.fun[0]]
    return np.array(w_matrix), np.array(residual)
