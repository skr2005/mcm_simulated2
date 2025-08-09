from .prelude import *
from .parameters import calc_all_5_parameters
from .data.problem1 import SpectrumProblem1
from .data.problem2 import Problem2Data
from .problem2 import p2s1_solution, p2s2_solution


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


def main(): ...


if __name__ == "__main__":
    main()
