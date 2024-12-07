
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ssb_d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.712287875981217e-01, -5.994008843337764e-01, -3.628844354235388e-01, -1.360334332445940e-01, -7.396828839870323e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ssb_d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.953245737161813e-01, -6.165255774159832e-17, -7.902131117317668e-01, -4.966671962439034e-16, -3.937695073622063e-01, 5.258193310355112e-18, -1.426873583308196e-01, -9.744764822092572e-17, -9.856040266199177e-03, -8.417396923318161e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ssb_d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.209508078405096e-03, 0.000000000000000e+00, 0.000000000000000e+00, -6.232488490451530e-03, 0.000000000000000e+00, 0.000000000000000e+00, -2.048513980377864e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.365755241170408e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.123508852880772e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
