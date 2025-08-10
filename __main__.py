from .prelude import *
from .parameters import calc_all_5_parameters
from .data.problem1 import SpectrumProblem1
from .data.problem2 import Problem2Data
from .data.problem3 import Problem3Data
from .problem2 import p2s1_solution, p2s2_solution
from .problem3 import p3_plot_melDER, p3_solution


def p1():
    """
    问题一的答案
    """
    return calc_all_5_parameters(SpectrumProblem1())


def p2():
    """
    问题二的答案
    """
    p2_data = Problem2Data()
    return p2s1_solution(p2_data), p2s2_solution(p2_data)


def p3():
    """
    问题三的答案
    """
    p2_data = Problem2Data()
    p3_data = Problem3Data()
    # p3_plot_melDER(p3_data)
    p3_solution(p2_data, p3_data)


"""
>>> p1()
(np.float64(3903.038236202643), np.float64(-0.0010163494546743452), np.float64(106.07949307088788), np.float64(91.79094031660622), np.float64(0.6407427827369949))
>>> p2()
(Problem2Solution(weights=array([1.5052e-01, 1.7068e-01, 2.4089e-01, 9.6361e-03, 4.2827e-01]), combined=<mcm_simulated2.problem2.CombinedSpectrum object at 0x0000018F9C4230E0>, CCT=np.float64(5500.000457716596), Duv=np.float64(0.006138235511558066), Rg=np.float64(102.65525173330582), Rf=np.float64(92.85323293382683), melDER=np.float64(0.8361969993522335)), Problem2Solution(weights=array([7.0011e-05, 1.1311e-01, 9.1099e-05, 8.8672e-01, 4.2728e-06]), combined=<mcm_simulated2.problem2.CombinedSpectrum object at 0x0000018F9C510F50>, CCT=np.float64(2500.009697319552), Duv=np.float64(0.0008625394591722684), Rg=np.float64(98.39437580701367), Rf=np.float64(88.69818183276624), melDER=np.float64(0.3668081750183712)))
"""


def main():
    p3()


if __name__ == "__main__":
    main()
