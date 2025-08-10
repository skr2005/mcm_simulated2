from .prelude import *
from .data.problem3 import Problem3Data
from .data_abstract.spectrum_abstract import Spectrum
from .parameters import calc_all_5_parameters

plt.rcParams["font.family"] = ["SimHei"]


def p3_plot_melDER(data3: Problem3Data):
    melDER_axis = []
    for spectrum in data3.spectra():
        stacked = np.vstack((spectrum.wavelength(), spectrum.spd()))
        melDER = spd_to_aopicDER(stacked)[:, -1]
        melDER_axis.append(melDER)
    x = [i for i in range(len(data3.time_info()))]
    plt.scatter(x, melDER_axis, s=4, color="blue")
    plt.legend()
    plt.xlabel("时间")
    plt.ylabel("mel-DER")
    plt.title("mel-DER随时间的变化趋势")
    plt.show()
