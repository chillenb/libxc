
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_b97_gga1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_gga1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.011267830458241e-01, -6.048413696585027e-01, -3.564005603020062e-01, -2.125433770181094e-01, -1.949606907205747e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_b97_gga1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_gga1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.388247983906468e-01, -2.027317797846326e-01, -8.175845983931066e-01, -2.341070377683710e-01, -4.491064361127229e-01, -2.288462493440688e-01, -3.464225300751082e-02, 8.606321792983110e-02, -2.585711688058233e-02, 5.964668874995752e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_b97_gga1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_gga1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.048983843187408e-01, 0.000000000000000e+00, -9.426835682655120e+20, 5.530481345776458e-03, 0.000000000000000e+00, -7.118007743266064e+20, -5.275507208009786e-02, 0.000000000000000e+00, -3.089297256857448e+20, -2.785026397970769e+01, 0.000000000000000e+00, 1.197130405831923e+20, -5.925792559389906e+01, 0.000000000000000e+00, 1.540924665042736e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
