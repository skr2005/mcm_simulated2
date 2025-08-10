from .prelude import *
from .data.problem3 import Problem3Data
from .data_abstract.spectrum_abstract import Spectrum
from .parameters import calc_all_5_parameters

plt.rcParams["font.family"] = ["SimHei"]


def p3_plot_melDER(data3: Problem3Data):
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


def p3_least_squares(data3: Problem3Data):
    x = np.array([i + 5.5 for i in range(len(data3.time_info()))], dtype=float)
    y = []
    for spectrum in data3.spectra():
        stacked = np.vstack((spectrum.wavelength(), spectrum.spd()))
        melDER = spd_to_aopicDER(stacked)[:, -1]
        y.append(melDER[0])
    y = np.array(y, dtype=float)

    def residuals(params, x, y):
        a, b, c, d = params
        model = np.polyval([a, b, c, d], x)
        return model - y  # 返回残差（拟合值 - 观测值）

    initial_params = [0.2, 0.4877, 8.3306, 0]

    result = least_squares(residuals, initial_params, args=(x, y))

    a1,b1,c1,d1 = result.x
    print(result.x)

    y_fit =  y_fit = np.polyval([a1,b1,c1,d1], x)
    plt.scatter(x, y)
    plt.plot(x, y_fit, "r-", label="拟合函数")
    plt.legend()
    plt.show()
