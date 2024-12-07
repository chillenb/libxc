
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ssb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.871167062762021e+00, -1.334203172828440e+00, -4.206537677214646e-01, -1.659713098286200e-01, -7.831759138568378e-02, -2.054145939410726e-02, -3.838589125278786e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ssb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.413858674198952e+00, -2.416336867741794e+00, -1.524250022866683e+00, -1.525902870184031e+00, -3.871488109038503e-01, -3.872423223601423e-01, -2.261172436105654e-01, -2.610109713817645e-02, -7.646888231196261e-02, -8.296427275733117e-04, -2.744008898901285e-02, -2.724372784170518e-02, -5.541556239928116e-04, -3.939543542357763e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ssb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.378373126963095e-04, 0.000000000000000e+00, -1.369474063356403e-04, -1.353069541537172e-03, 0.000000000000000e+00, -1.347383145144583e-03, -9.480740481542441e-02, 0.000000000000000e+00, -9.466954657244422e-02, 2.302350713241714e+00, 0.000000000000000e+00, -3.967298811868181e-01, -6.118378279723761e+01, 0.000000000000000e+00, -2.403930104917542e+00, -4.027054660267887e-01, 0.000000000000000e+00, -3.777484784592750e-01, -1.756850002646122e+00, 0.000000000000000e+00, -2.503849358375010e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
