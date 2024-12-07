
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_2d_pbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.230569416133227e-01, -7.652952371440420e-01, -3.963380037519212e-01, -6.262391716396372e-02, -6.429787969169861e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_2d_pbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.229843915167411e+00, 3.019274741233773e-16, -9.874388012817882e-01, -1.213321386973674e-16, -4.900543720774736e-01, -1.569226258590742e-16, -9.317438467688849e-02, -6.594789606526316e-18, -9.644675863475666e-04, 1.436338904088452e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_2d_pbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.464745835943445e-02, 0.000000000000000e+00, 0.000000000000000e+00, -6.626890281566572e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.027712126316573e-01, 0.000000000000000e+00, 0.000000000000000e+00, -7.618968828607703e-02, 0.000000000000000e+00, 0.000000000000000e+00, -4.405869986044167e-04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
