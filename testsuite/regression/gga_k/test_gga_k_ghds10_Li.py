
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ghds10_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.152143424457730e+01, 1.300716264415231e+01, 4.218396178812181e+00, 4.129227577192973e-01, -9.777276152496937e-02, 2.065283327872815e+00, -1.614753878786764e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ghds10_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.380297573513403e+01, 2.385113978788430e+01, 1.005891804661253e+01, 1.007885440564632e+01, -1.751064906975095e+00, -1.761485005583675e+00, 6.216794388985041e-01, -3.935723422467374e+00, -6.450619485721229e-02, -3.777437945674814e+00, -3.886108281196990e+00, -3.995501017138625e+00, -4.182995677640515e+00, -4.116987761580632e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ghds10_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.927193223082906e-02, 0.000000000000000e+00, 1.921993237398214e-02, 5.758341797473614e-02, 0.000000000000000e+00, 5.743663279313743e-02, 4.153406735139716e+00, 0.000000000000000e+00, 4.158710224823615e+00, 2.605032494947879e+01, 0.000000000000000e+00, 7.829811571772349e+04, 4.181517806762772e+02, 0.000000000000000e+00, 2.454376727176486e+09, 6.733474755479385e+04, 0.000000000000000e+00, 6.882557364072799e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375079883e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
