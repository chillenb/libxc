
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_yt65_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_yt65", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.670957723474534e+01, 8.467727675036850e+00, 1.083952860478673e+00, 1.338385284193022e-01, 3.470882191321702e-02, 6.175779882314061e-01, 2.713797168333975e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_yt65_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_yt65", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.557617287618726e+01, 2.562382248175994e+01, 1.191563459236996e+01, 1.193688945944512e+01, 9.278273571021627e-02, 9.024031812603318e-02, 2.120915686048689e-01, -6.103754937149158e-01, 1.962976097527561e-02, -2.420444367493600e-01, -6.052822605914467e-01, -6.264602049763210e-01, -2.836889519453260e-01, -2.371170822073742e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_yt65_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_yt65", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.854386446165813e-03, 0.000000000000000e+00, 3.843986474796428e-03, 1.151668359494723e-02, 0.000000000000000e+00, 1.148732655862749e-02, 8.306813470279432e-01, 0.000000000000000e+00, 8.317420449647228e-01, 5.210064989895757e+00, 0.000000000000000e+00, 1.565962314354470e+04, 8.363035613525544e+01, 0.000000000000000e+00, 4.908753454352971e+08, 1.346694951095877e+04, 0.000000000000000e+00, 1.376511472814560e+04, 1.647228622999552e+09, 0.000000000000000e+00, 4.584716750159767e+09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
