
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_absp2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.650393357367848e-01, -7.333989192074310e-02, 3.133888081875173e-01, 4.952039449874835e-01, 6.849691065477987e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_absp2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.135734325152334e+00, -1.007895727358300e-16, -1.374436153038153e+00, -9.624681836446353e-18, -8.086778445156808e-01, 6.199264315526150e-17, -5.305277867828498e-01, 2.999896778509576e-16, -6.850475123030713e-01, -1.579516509208668e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_absp2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.192899183611812e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.816356732758560e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.906799273274511e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.526164046858387e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459646400083933e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
