
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_htbs_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220345560888795e-01, -5.762934998000241e-01, -3.546231863261353e-01, -1.483821072862778e-01, -7.399215932413357e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_htbs_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282835685056642e-01, -5.518218033960693e-17, -7.230249718991756e-01, -2.118271600608641e-16, -4.139791840504748e-01, -3.140719371206751e-17, -1.495290601940441e-01, -7.892043597454807e-17, -9.865621358420359e-03, -2.654633648889381e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_htbs_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.680875200637492e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.107221754394991e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.285282718415097e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.438193398581639e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
