
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.037422781146677e+00, -1.413111899517623e+00, -3.137142327630846e-01, -1.841757828165247e-01, -7.184056513435093e-02, -5.414761666731365e-03, -2.726525612351702e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_scanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.719573523920550e+00, -2.722014612766085e+00, -1.892035111043257e+00, -1.893636405881253e+00, -3.058368236096024e-02, -1.282803134249173e-01, -2.457883112651349e-01, -8.918881415008737e-03, -9.908290499170663e-02, -8.755920524073625e-05, -1.031539938774330e-02, -9.393756392410555e-03, -6.805048140676972e-04, 8.716796003324746e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.260008312118206e-06, 0.000000000000000e+00, 7.224577549782116e-06, 4.578689914843153e-05, 0.000000000000000e+00, 4.561034488914357e-05, -4.846302730585352e-01, 0.000000000000000e+00, -3.640251142495684e-01, 6.865366266803088e-02, 0.000000000000000e+00, 2.241383872417815e+01, 7.211252344146639e+00, 0.000000000000000e+00, 2.101973126344902e+04, 2.707653363110932e+01, 0.000000000000000e+00, 2.011979324197346e+01, 4.131837043267923e+04, 0.000000000000000e+00, -3.035811280178675e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 2.266999779489386e-02, 1.714593344158554e-02, 0.000000000000000e+00, -3.355006520254705e-07, 0.000000000000000e+00, 7.870582815199981e-09, -4.089448670344404e-07, -3.252739533243844e-07, -1.953553292073623e-15, 4.304229707291064e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
