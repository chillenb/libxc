
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.833640870733615e+00, -1.287885778807687e+00, -4.725759945629842e-01, -1.652478579200030e-01, -8.352300996006172e-02, -2.947903197691907e-02, -5.511246082170061e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.403058025908846e+00, -2.405322271852733e+00, -1.618875811342054e+00, -1.620389332760402e+00, -2.841992294366668e-01, -2.845345351174708e-01, -2.190540659244443e-01, -3.743362790170254e-02, -5.680711348648075e-02, -1.191154853320638e-03, -3.934902512032303e-02, -3.906972263966499e-02, -7.956271700140153e-04, -5.656189440363077e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.976515472690313e-05, 0.000000000000000e+00, -6.940851710675734e-05, -5.142011908746719e-04, 0.000000000000000e+00, -5.120427268951742e-04, -1.676888858893524e-01, 0.000000000000000e+00, -1.674318069563622e-01, -6.506741479397752e-01, 0.000000000000000e+00, -7.956571627406086e-01, -1.193832398858256e+02, 0.000000000000000e+00, -5.091089305303210e+00, -8.085286542078429e-01, 0.000000000000000e+00, -7.550354960828610e-01, -3.706127104536785e+00, 0.000000000000000e+00, -5.304940280525777e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
