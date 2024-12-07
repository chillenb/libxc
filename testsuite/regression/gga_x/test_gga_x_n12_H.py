
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_n12_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.221004149491457e-01, -5.816351895935346e-01, -3.405906547690763e-01, -9.051559286690884e-02, -6.427995378724340e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_n12_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.313748391926901e-01, 3.536511320650004e-18, -7.305975029677089e-01, -2.162831137185572e-17, -4.132273163257054e-01, -8.275811281716315e-17, -2.310210249032323e-01, -1.769089762453965e-17, -8.253280387592444e-03, -1.880695540963081e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_n12_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.716305784293140e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.641936486880394e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.188601207625271e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.294076735605458e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.455231160327429e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
