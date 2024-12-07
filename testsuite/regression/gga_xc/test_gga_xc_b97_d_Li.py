
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_b97_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.909859433511796e+00, -1.344456779430025e+00, -4.723430211770033e-01, -1.675647951310278e-01, -8.558050681128926e-02, -1.956118289078445e-02, -4.112155329813335e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_b97_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.521711934995198e+00, -2.523989866502515e+00, -1.727782318198439e+00, -1.729280009233815e+00, -2.796045200101603e-01, -2.804323126861537e-01, -2.214490068393271e-01, 8.807770922520125e-01, -4.836741604605811e-02, 5.669178779180125e-01, -2.788520556733707e-02, -2.654010771443865e-02, -9.308228831123842e-04, 5.152418757517257e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_b97_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.059207449892398e-05, 0.000000000000000e+00, -2.020461430861965e-05, -2.659296753066998e-04, 0.000000000000000e+00, -2.639404539726300e-04, -1.710692183174398e-01, 0.000000000000000e+00, -1.707251854976897e-01, -1.248777322832204e+00, 0.000000000000000e+00, 1.350962662659962e+02, -1.421400342033209e+02, 0.000000000000000e+00, 1.618436379304908e+04, -6.915730223413111e-01, 0.000000000000000e+00, -4.351055177447340e-01, -8.063040117897909e+00, 0.000000000000000e+00, 2.732234626840529e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
