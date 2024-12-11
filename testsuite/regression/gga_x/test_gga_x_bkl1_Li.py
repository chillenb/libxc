
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bkl1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.876644937301506e+00, -1.391740579347255e+00, -5.031907276812433e-01, -1.644655628630539e-01, -9.934798706396523e-02, -1.140077670557988e-02, -2.127820881496785e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bkl1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.146538294498819e+00, -2.148725027400396e+00, -1.410770220178905e+00, -1.412116574225863e+00, -5.862225432855011e-01, -5.868593217214197e-01, -1.998127881802469e-01, -1.456330158763211e-02, -8.861440290380232e-02, -4.598929181543446e-04, -1.538472796530922e-02, -1.523618524372863e-02, -3.071820507310970e-04, -2.183783727950879e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bkl1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.077946296731732e-04, 0.000000000000000e+00, -6.057297435071988e-04, -2.340314601948719e-03, 0.000000000000000e+00, -2.332967352972365e-03, -4.111410036634876e-02, 0.000000000000000e+00, -4.067579475025521e-02, -9.509152734975896e+00, 0.000000000000000e+00, 5.667615016924635e-01, -9.595313908098437e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.061770873831979e+00, 0.000000000000000e+00, 7.615168342315431e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
