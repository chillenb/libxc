
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_tpsslyp1w_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.127008273171961e-01, -5.699251656317821e-01, -3.325980889945436e-01, 2.992418926428603e+165, 1.582997964255450e+151]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_tpsslyp1w_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.388240621693895e-01, -3.548514595866504e-01, -7.637544682095232e-01, -2.315363054760823e-01, -4.438181455086975e-01, -1.678313025009752e-01, -1.500361493122503e+165, 4.146754663455144e+170, 2.820584164837298e+152, 6.381830493968987e+152]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.483026406978011e+00, 2.164366886827692e-02, 1.622653450346063e-02, -3.044507794620998e-02, 3.438719061436823e-02, 2.575692157951031e-02, -1.354072521519714e-01, 2.950210920792956e-01, 2.212592819169965e-01, -3.581036641591479e+00, 1.260062407992866e+01, 9.450452070551076e+00, -3.964838492439620e+05, 5.296779305705816e-18, 3.972578537740708e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.068800960210618e+01, 0.000000000000000e+00, 7.073681261703720e-03, 0.000000000000000e+00, 1.940700615598679e-03, 0.000000000000000e+00, -6.240536466219934e-04, 0.000000000000000e+00, -2.586912629106759e-08, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
