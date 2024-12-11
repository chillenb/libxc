
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_gam_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.040127856155150e-03, 2.316263307997496e-02, 5.853118396848762e-03, -1.293974997001051e-02, -1.765781506186283e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_gam_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.100735666222464e-03, -2.292937200581764e-01, 2.252681263055080e-02, -1.989748742302159e-01, 5.647305898675421e-02, -1.398146316750259e-01, -1.020692743045740e-02, -4.598194406602432e-02, -2.250512549272819e-03, 1.127446801287970e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_gam_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.417190885544092e-02, 0.000000000000000e+00, 4.768093113053344e+20, 1.643523409346723e-03, 0.000000000000000e+00, 2.930728693411322e+20, -1.087554226773961e-01, 0.000000000000000e+00, 3.589819880617092e+19, -5.758188195555760e-01, 0.000000000000000e+00, 9.796064115427749e+19, -8.587693195287981e-01, 0.000000000000000e+00, 5.306211026764002e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
