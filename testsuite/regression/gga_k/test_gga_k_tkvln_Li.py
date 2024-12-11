
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tkvln_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tkvln", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.014776318611663e+01, 1.180372484555195e+01, 3.685930283221173e+00, 1.712002843412043e-01, 1.020903331768478e-01, 3.085315680126904e+00, 1.381109146277941e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tkvln_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tkvln", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.227425481242681e+01, 2.232200062571371e+01, 8.699913016185844e+00, 8.719452988678178e+00, -2.441376952518804e+00, -2.451596344096557e+00, 2.209346290863988e-01, -3.055270580803842e+00, -2.452266411884438e-02, -1.196715684860382e+00, -3.030062843352012e+00, -3.135912929297372e+00, -1.400206367184505e+00, -1.162818122861988e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tkvln_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tkvln", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.927193223082906e-02, 0.000000000000000e+00, 1.921993237398214e-02, 5.758341797473614e-02, 0.000000000000000e+00, 5.743663279313743e-02, 4.153406735139716e+00, 0.000000000000000e+00, 4.158710224823615e+00, 2.605032494947879e+01, 0.000000000000000e+00, 7.829811571772349e+04, 4.181517806762772e+02, 0.000000000000000e+00, 2.454376727176486e+09, 6.733474755479385e+04, 0.000000000000000e+00, 6.882557364072799e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375079883e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
