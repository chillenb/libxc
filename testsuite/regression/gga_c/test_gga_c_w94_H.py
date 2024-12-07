
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_w94_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_w94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.622853675052743e-08, -2.029605060118539e-08, -1.650406304169515e-08, -2.313855547777572e-09, -8.441906872433834e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_w94_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_w94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.386307290629896e-08, -3.913943947519907e+05, -3.941174178277934e-08, -2.184116748497144e+05, -6.033612296655815e-08, -3.549047040290054e+04, -1.086951215750709e-08, -9.475765158088308e+01, -3.999830036220157e-11, -3.614709587377355e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_w94_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_w94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.614004474557481e-09, 3.228008949114962e-09, 1.614004474557481e-09, 1.339529548250073e-08, 2.679059096500146e-08, 1.339529548250073e-08, 1.131197650412685e-07, 2.262395300825371e-07, 1.131197650412685e-07, 1.091316743669296e-06, 2.182633487338593e-06, 1.091316743669296e-06, 2.856744229922987e-05, 5.713488459845975e-05, 2.856744229922987e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
