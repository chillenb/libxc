
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optb88_vdw_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb88_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220345781692855e-01, -5.758454264633859e-01, -3.549275159400648e-01, -1.374186217063766e-01, -5.167642797406917e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optb88_vdw_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb88_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282857730555249e-01, -1.201575408401664e-16, -7.232038380293585e-01, -2.300731048094664e-16, -4.094665659347474e-01, 5.194476356185415e-17, -1.046881087634857e-01, -9.138482309398608e-17, -1.374813733716447e-02, -6.795931948175323e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optb88_vdw_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb88_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.677538866314080e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.071163529712246e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.392697109756988e-01, 0.000000000000000e+00, 0.000000000000000e+00, -8.840089569816076e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.407217486993239e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
