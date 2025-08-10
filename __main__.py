from .prelude import *
from .parameters import calc_all_5_parameters
from .data.problem1 import SpectrumProblem1
from .data.problem2 import Problem2Data
from .data.problem3 import Problem3Data
from .problem2 import p2s1_solution, p2s2_solution
from .problem3 import p3_melDERs, solve_problem3_weights, draw_comparison


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

    调用过程会绘制三个时间点的SPD对比图
    """
    p2_data = Problem2Data()
    p3_data = Problem3Data()
    w_matrix, residual = solve_problem3_weights(
        p2_data.five_leds(), p3_melDERs(p3_data)
    )
    draw_comparison(w_matrix, p2_data.five_leds(), p3_data)
    return w_matrix, residual


"""
>>> p1()
(np.float64(3903.038236202643), np.float64(-0.0010163494546743452), np.float64(106.07949307088788), np.float64(91.79094031660622), np.float64(0.6407427827369949))
>>> p2()
(Problem2Solution(weights=array([1.5052e-01, 1.7068e-01, 2.4089e-01, 9.6361e-03, 4.2827e-01]), combined=<mcm_simulated2.problem2.CombinedSpectrum object at 0x0000018F9C4230E0>, CCT=np.float64(5500.000457716596), Duv=np.float64(0.006138235511558066), Rg=np.float64(102.65525173330582), Rf=np.float64(92.85323293382683), melDER=np.float64(0.8361969993522335)), Problem2Solution(weights=array([7.0011e-05, 1.1311e-01, 9.1099e-05, 8.8672e-01, 4.2728e-06]), combined=<mcm_simulated2.problem2.CombinedSpectrum object at 0x0000018F9C510F50>, CCT=np.float64(2500.009697319552), Duv=np.float64(0.0008625394591722684), Rg=np.float64(98.39437580701367), Rf=np.float64(88.69818183276624), melDER=np.float64(0.3668081750183712)))
>>> p3()
(array([[2.7479e-01, 2.3391e-01, 6.1116e-02, 3.3560e-02, 3.9663e-01],
       [2.7399e-01, 2.3335e-01, 6.1444e-02, 3.5576e-02, 3.9565e-01],
       [2.7316e-01, 2.3277e-01, 6.1784e-02, 3.7652e-02, 3.9464e-01],
       [2.7231e-01, 2.3216e-01, 6.2137e-02, 3.9790e-02, 3.9360e-01],
       [2.8976e-01, 2.2870e-01, 5.9251e-02, 3.4115e-02, 3.8818e-01],
       [3.2218e-01, 2.1840e-01, 5.7472e-02, 3.3113e-02, 3.6884e-01],
       [3.3482e-01, 2.1597e-01, 5.5464e-02, 2.9036e-02, 3.6470e-01],
       [3.4648e-01, 2.1364e-01, 5.3662e-02, 2.5647e-02, 3.6058e-01],
       [3.4375e-01, 2.0451e-01, 5.7508e-02, 3.0097e-02, 3.6414e-01],
       [3.3686e-01, 2.0681e-01, 6.1591e-02, 3.4288e-02, 3.6046e-01],
       [3.2870e-01, 2.0793e-01, 6.5829e-02, 3.9046e-02, 3.5849e-01],
       [3.2509e-01, 2.0611e-01, 6.7009e-02, 4.6556e-02, 3.5523e-01],
       [3.2047e-01, 2.0379e-01, 6.8808e-02, 5.6209e-02, 3.5072e-01],
       [3.1527e-01, 2.0377e-01, 8.7065e-02, 7.7370e-02, 3.1652e-01],
       [2.3366e-01, 1.5628e-01, 1.2184e-01, 2.5012e-01, 2.3810e-01]]), array([-1.3160e-07, 9.8562e-10, 1.1010e-09, 1.2317e-09, -5.2072e-10,
       -1.1504e-08, -2.5781e-11, -1.2110e-11, 3.7091e-09, 1.5425e-11,
       9.0867e-10, 1.3600e-13, 2.9735e-09, -3.2562e-08, 1.6233e-11]))
"""
