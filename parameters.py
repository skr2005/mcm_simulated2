from .prelude import *
from .data_abstract.spectrum_abstract import Spectrum


def calc_all_5_parameters(
    spectrum: Spectrum,
) -> tuple[
    np.float64,
    np.float64,
    np.float64,
    np.float64,
    np.float64,
]:
    """
    根据光谱计算出：
    CCT, Duv, Rg, Rf, melDER
    """

    stacked = np.vstack((spectrum.wavelength(), spectrum.spd()))

    Rf, Rg, CCT, Duv = cast(Any, spd_to_iesrf(stacked, "Rf,Rg,cct,duv"))

    melDER = spd_to_aopicDER(stacked)[:, -1]

    return CCT[0][0], Duv[0][0], Rg[0][0], Rf[0][0], melDER[0]
