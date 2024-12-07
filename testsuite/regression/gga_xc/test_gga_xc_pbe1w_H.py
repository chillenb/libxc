
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_pbe1w_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbe1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.541876287320384e-01, -5.990514406208665e-01, -3.719488628054536e-01, -1.379584151434873e-01, -7.802303507891904e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_pbe1w_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbe1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.651035050156042e-01, 1.111520109754858e+00, -7.582904995042848e-01, 5.460022750015710e+01, -4.293480942405072e-01, 3.057999290903197e+01, -1.430913823905196e-01, 2.258845269464316e-01, -1.037362044638723e-02, -1.808885943039938e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_pbe1w_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbe1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.683714222772004e-03, 2.433161848960595e-02, 1.216580924480297e-02, -1.626421352964929e-02, 1.510623062421261e-02, 7.553115312106305e-03, -1.386871942254738e-01, 6.228871760927827e-02, 3.114435880463913e-02, -4.535725701045211e+00, 1.304037471549920e-01, 6.520187357749566e-02, -5.535645963674826e+00, 1.469234439303380e-03, 7.346172199867997e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
