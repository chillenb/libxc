
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rppscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.974920765872153e+00, -1.311268315472403e+00, -2.342832281820834e-01, -1.809544674590481e-01, -5.246927185872651e-02, -4.816589062986531e-03, -1.021140385176983e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rppscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.757308827267502e+00, -2.759876991369624e+00, -1.960708310085863e+00, -1.962190829457346e+00, -3.254056935755857e-01, -3.259033195562962e-01, -2.474325994771338e-01, 3.100464165747890e-01, -7.766451651481042e-02, 3.397467277111611e-03, 1.841834238653130e-01, 3.235860568481085e-01, 1.015185605952393e-04, -8.607518337074510e-23]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rppscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.786737385428936e-04, 0.000000000000000e+00, -3.772534715301327e-04, -2.149952155488768e-03, 0.000000000000000e+00, -2.140886740413116e-03, -4.332414125240335e-01, 0.000000000000000e+00, -4.337385111893367e-01, -4.798477981625031e+00, 0.000000000000000e+00, -8.200972651164244e+03, -2.002628693891181e+02, 0.000000000000000e+00, -7.087792219040166e+06, -8.601647591997569e+01, 0.000000000000000e+00, -7.331944745012736e+03, -4.803770781739178e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rppscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.786242651857874e-02, 1.784441111538341e-02, 3.118375243230420e-02, 3.114554499311822e-02, 2.378177799997488e-03, 2.508153711235688e-03, 1.730428248483909e-01, 1.050986093253013e-01, 6.451770599445762e-02, 2.898664949313307e-03, 1.281996041267831e-03, 1.068954787883535e-01, 5.856115733939118e-10, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
