from numpy import dtype
from .prelude import *
from .data.problem2 import Problem2Data
from .data.problem3 import Problem3Data
from .problem2 import combine_spd
from .data_abstract.spectrum_abstract import Spectrum
from .parameters import calc_all_5_parameters

plt.rcParams["font.family"] = ["SimHei"]


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


def p3_solution(data2: Problem2Data, data3: Problem3Data):
    """
    使用最小二乘法衡量拟合效果
    """
    # y为目标mel-DER
    y_target = []
    arr = []
    for spectrum in data3.spectra():
        stacked = np.vstack((spectrum.wavelength(), spectrum.spd()))
        arr.append(spectrum.spd())
        melDER = spd_to_aopicDER(stacked)[:, -1]
        y_target.append(melDER[0])
    y_target = np.array(y_target, dtype=float)
    arr = np.array(arr, dtype=float)

    # 五个leds的spectrum
    leds = []
    for spectrum in data2.five_leds():
        leds.append(spectrum.spd())
    leds = np.array(leds, dtype=float)

    wl = data2.five_leds()[0].wavelength()

    for i in range(len(arr)):

        def residuals(w):
            w = np.exp(w)
            w = w / w.sum()
            spd = combine_spd(data2.five_leds(), w)
            res = spd - arr[i]
            return res.flatten()

        w = np.zeros(5)

        result = least_squares(residuals, w)
        result.x = np.exp(result.x)
        result.x = result.x / result.x.sum()
        print(result.x)
