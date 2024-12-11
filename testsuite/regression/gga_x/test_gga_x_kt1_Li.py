
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_kt1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.843755423943394e+00, -1.352074392299712e+00, -3.318361402690172e-01, -1.569470396514966e-01, -6.222889480326720e-02, -1.139787943111726e-02, -2.127820965719286e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_kt1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.171446899399965e+00, -2.173654605383704e+00, -1.417550190365672e+00, -1.418934238224034e+00, -3.810834614508147e-01, -3.809179113149839e-01, -2.092658665834750e-01, -1.450062406183421e-02, -8.295816410288241e-02, -4.598929181543212e-04, -1.524839211734565e-02, -1.513748933865591e-02, -3.071820507310971e-04, -2.183783727950879e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_kt1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.919594560353022e-04, 0.000000000000000e+00, -4.902048214802675e-04, -2.061349402651428e-03, 0.000000000000000e+00, -2.054586108301680e-03, -5.486396554400828e-02, 0.000000000000000e+00, -5.487195057375636e-02, -5.951829666719937e-02, 0.000000000000000e+00, -5.999998880484111e-02, -5.998800963135268e-02, 0.000000000000000e+00, -5.999999992242971e-02, -5.999998631077526e-02, 0.000000000000000e+00, -5.999998670470278e-02, -5.999999999999771e-02, 0.000000000000000e+00, -5.999999999999940e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
