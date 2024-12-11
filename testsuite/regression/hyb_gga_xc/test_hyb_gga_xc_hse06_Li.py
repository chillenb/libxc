
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hse06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.417782428260779e+00, -1.021708241272463e+00, -3.309748979872625e-01, -1.484541525075943e-01, -7.424805558970921e-02, -2.045636684024764e-02, -3.838584770558010e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hse06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.804919539800875e+00, -1.806421365721237e+00, -1.238883981980082e+00, -1.239817163680255e+00, -3.308105597921966e-01, -3.310329264552972e-01, -1.917534523093406e-01, -1.236116060912498e-01, -7.552122232107271e-02, 3.419889397264876e-01, -2.728152086727680e-02, -2.708750813878819e-02, -5.541554329784571e-04, -3.939542297868219e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hse06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.521405790199178e-04, 9.190971700708733e-05, -1.514036650475776e-04, -7.005890331699740e-04, 2.980993506782570e-04, -6.978176012023004e-04, -5.374471388316948e-02, 6.249948659585063e-03, -5.356566177524297e-02, 2.553500803631379e-01, 6.762268918356340e+00, 3.381134459178170e+00, -4.447739325090438e+01, 2.258698854598489e+01, 1.129349427299244e+01, 1.678587300264123e-04, 3.357174600576258e-04, 1.678587300264123e-04, 1.606543586949356e-06, 3.212885779437900e-06, 1.606543586949356e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
