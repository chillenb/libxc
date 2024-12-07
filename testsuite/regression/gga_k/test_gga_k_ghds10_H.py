
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ghds10_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.979969605369012e+00, 2.987230847451244e+00, 1.678664564617834e+00, 5.228311213584860e-01, -8.344527836686746e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ghds10_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [4.478472293893468e+00, 3.447506658946765e-16, 3.301271402512501e+00, 5.706663397366814e-16, 1.216378782689595e+00, 1.880065105535239e-16, -3.044703715032122e-01, 2.709829965638825e-16, -2.041390786367502e+00, 7.289382042859648e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ghds10_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.192899183611812e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.816356732758560e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.906799273274511e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.526164046858387e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459646400083933e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
