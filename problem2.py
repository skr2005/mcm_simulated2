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


"""
>>> scenario_1(Problem2Data())
             message: Optimization terminated successfully.
             success: True
                 fun: -92.85323313874815
                   x: [ 1.505e-01  1.707e-01  2.409e-01  9.636e-03
                        4.283e-01]
                 nit: 517
                nfev: 1596
          population: [[ 1.491e-01  1.701e-01 ...  7.554e-03  4.287e-01]
                       [ 1.586e-01  1.708e-01 ...  1.441e-02  4.237e-01]
                       ...
                       [ 1.457e-01  1.646e-01 ...  4.404e-03  4.382e-01]
                       [ 1.455e-01  1.640e-01 ...  5.220e-03  4.406e-01]]
 population_energies: [-9.285e+01 -9.284e+01 ... -9.285e+01 -9.284e+01]
              constr: [array([ 0.000e+00,  0.000e+00,  0.000e+00])]
    constr_violation: 0.0
               maxcv: 0.0
                 jac: [array([[ 1.000e+00,  1.000e+00, ...,  1.000e+00,
                               1.000e+00],
                             [ 3.175e+03,  3.573e+03, ..., -1.293e+04,
                               7.302e+02],
                             [ 2.583e+01, -3.068e+01, ...,  2.121e+01,
                              -1.517e+01]], shape=(3, 5)), array([[ 1.000e+00,  0.000e+00, ...,  0.000e+00,      
                               0.000e+00],
                             [ 0.000e+00,  1.000e+00, ...,  0.000e+00,
                               0.000e+00],
                             ...,
                             [ 0.000e+00,  0.000e+00, ...,  1.000e+00,
                               0.000e+00],
                             [ 0.000e+00,  0.000e+00, ...,  0.000e+00,
                               1.000e+00]], shape=(5, 5))]
>>> _.x
array([1.5052e-01, 1.7068e-01, 2.4089e-01, 9.6356e-03, 4.2827e-01])
"""


def scenario_2(problem2_data: Problem2Data) -> OptimizeResult:
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

    def combine(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        组合光谱，没变
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
        melDER = spd_to_aopicDER(stacked)[:, -1]
        return melDER[0]

    def constraints_fn(
        weights: NDArray[np.float64],
    ) -> tuple[np.float64, np.float64, np.float64]:
        """
        计算权值和、CCT、Rf
        """
        stacked = np.vstack((anyone_wavelength, combine(weights)))
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


"""
>>> scenario_2(Problem2Data())
             message: Optimization terminated successfully.
             success: True
                 fun: 0.36680482342085124
                   x: [ 4.326e-06  1.131e-01  2.957e-05  8.868e-01
                        1.401e-05]
                 nit: 802
                nfev: 4821
          population: [[ 4.326e-06  1.131e-01 ...  8.868e-01  1.401e-05]
                       [ 1.915e-04  1.131e-01 ...  8.863e-01  2.495e-05]
                       ...
                       [ 4.833e-05  1.132e-01 ...  8.866e-01  1.888e-05]
                       [ 1.220e-04  1.131e-01 ...  8.867e-01  2.234e-05]]
 population_energies: [ 3.668e-01  3.668e-01 ...  3.668e-01  3.668e-01]
              constr: [array([ 0.000e+00,  0.000e+00,  0.000e+00])]
    constr_violation: 0.0
               maxcv: 0.0
>>> _.x
array([4.3256e-06, 1.1312e-01, 2.9568e-05, 8.8683e-01, 1.4012e-05])
"""


def q2_params_output():
    """
    问题二中所需的五个参数的计算和输出
    """
    weights1 = np.array([1.505e-01, 1.707e-01, 2.409e-01, 9.636e-03, 4.283e-01])
    weights2 = np.array([4.326e-06, 1.131e-01, 2.957e-05, 8.868e-01, 1.401e-05])
    problem2_data = Problem2Data()
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

    spd1 = combine(weights1)
    spd2 = combine(weights2)
    stacked1 = np.vstack((anyone_wavelength, spd1))
    stacked2 = np.vstack((anyone_wavelength, spd2))
    Rf1, Rg1, CCT1, Duv1 = cast(Any, spd_to_iesrf(stacked1, "Rf,Rg,cct,duv"))
    Rf2, Rg2, CCT2, Duv2 = cast(Any, spd_to_iesrf(stacked2, "Rf,Rg,cct,duv"))
    melDER1 = spd_to_aopicDER(stacked1)[:, -1]
    melDER2 = spd_to_aopicDER(stacked2)[:, -1]
    print(f"场景一数据:Rf1:{Rf1},Rg1:{Rg1},CCT1{CCT1},Duv1{Duv1},mel-DER1={melDER1}")
    print(f"场景二数据:Rf2:{Rf2},Rg1:{Rg2},CCT1{CCT2},Duv1{Duv2},mel-DER2={melDER2}")
