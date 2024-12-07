
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_sogga11_x_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.721795516812271e-01, -3.392791553632019e-01, -2.025204753404654e-01, -9.084250253531457e-03, 1.130201922920054e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_sogga11_x_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.958721273380851e-01, -3.218011706956949e-17, -4.402081875421344e-01, -1.220233653791746e-16, -2.580366276042021e-01, -3.268728256710198e-18, -1.068732068474475e-01, -5.567531153268453e-18, 3.928303826676645e-06, 1.596103341298406e-24]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_sogga11_x_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.633518536491397e-03, 0.000000000000000e+00, 0.000000000000000e+00, -5.650064514961787e-03, 0.000000000000000e+00, 0.000000000000000e+00, -2.618683311557105e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.066627947647439e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.934862385703370e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
