
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_pbelyp1w_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbelyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.304604936874841e-01, -5.855991567493366e-01, -3.659641941174847e-01, -1.378410497006417e-01, -7.802303547498311e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_pbelyp1w_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbelyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.377399814698989e-01, -2.450141668596068e-01, -7.278407607716337e-01, -2.519781966670389e-01, -4.089278094369830e-01, -1.978538093371852e-01, -1.423939288962579e-01, -5.076325836282431e-02, -1.037361939447153e-02, -3.624513323217621e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_pbelyp1w_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbelyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.684952346757553e-02, 2.164366886827692e-02, 1.622653450346063e-02, -2.381732884175677e-02, 3.438719061436823e-02, 2.575692157951031e-02, -1.698315530301535e-01, 2.950210920792956e-01, 2.212592819169965e-01, -4.600927574597276e+00, 1.260062407992866e+01, 9.450452070551076e+00, -5.536380580894813e+00, 5.296779305705816e-18, 3.972578537740708e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
