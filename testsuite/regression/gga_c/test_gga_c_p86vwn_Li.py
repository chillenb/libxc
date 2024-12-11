
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_p86vwn_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.412858422859917e-02, -4.768936266902273e-02, 3.875086737792903e-03, -1.577801434614419e-02, -2.432412263493292e-03, -6.794960882900992e-03, -1.629259198827822e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_p86vwn_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.178415980422796e-01, -1.177241821062459e-01, -1.051736600517363e-01, -1.050868102288706e-01, -2.448387347899375e-02, -2.448826637538694e-02, -2.327938744527257e-02, -1.293580889603090e-01, -1.432571187171524e-02, -6.848427264468007e-02, -8.545028099609225e-03, -8.639536931595125e-03, -1.914214508247163e-04, -2.832624807333714e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_p86vwn_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.363533738525183e-05, 8.727067477050367e-05, 4.363533738525183e-05, 1.467783819468145e-04, 2.935567638936289e-04, 1.467783819468145e-04, 6.723580691440459e-03, 1.344716138288092e-02, 6.723580691440459e-03, 2.622330052275132e+00, 5.244660104550263e+00, 2.622330052275132e+00, 2.919550257118503e+01, 5.839100514237006e+01, 2.919550257118503e+01, -4.791672345776499e-03, -9.583344691552998e-03, -4.791672345776499e-03, -1.049678891068065e-29, -2.099357782136131e-29, -1.049678891068065e-29])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
