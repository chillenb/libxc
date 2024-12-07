
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sfat_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.548209374729463e-01, -4.077197967226548e-01, -2.002662129919359e-01, -1.865458722705558e-02, -2.778958719492594e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sfat_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.495426094597196e-01, -1.559316652717341e-17, -5.555388160260366e-01, -1.123067787975329e-16, -2.706264525053120e-01, -3.588711115871556e-17, -3.081601370676993e-02, -8.761279786311289e-18, -5.557306053650322e-06, 1.930105278631203e-22]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sfat_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.306558769092527e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.496684683635886e-02, 0.000000000000000e+00, 0.000000000000000e+00, -6.888230160786743e-02, 0.000000000000000e+00, 0.000000000000000e+00, -3.537944807660415e-01, 0.000000000000000e+00, 0.000000000000000e+00, -3.019301692083137e-04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
