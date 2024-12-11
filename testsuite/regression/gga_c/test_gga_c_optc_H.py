
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_optc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_optc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.121002234795571e-02, -1.260358215619668e-02, -5.966698706883086e-03, -1.211487980423536e-04, 9.607769798437050e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_optc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_optc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.450993951118855e-02, 1.861655907968832e+00, -2.666966027324001e-02, 8.452696040579971e+01, -1.881662214460571e-02, 5.299257880781836e+01, -7.181907248908344e-04, 4.947795619904936e-01, -1.018062452484192e-09, 2.901541281061940e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_optc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_optc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.175916006292868e-02, 3.910253472737502e-02, -1.554749370019664e+13, 6.125928825033083e-03, 2.037044643383175e-02, -1.554796746927731e+13, 2.728586081980500e-02, 9.073320811262457e-02, -1.554780661812901e+13, 6.661762418088835e-02, 2.215230746819903e-01, -1.554785154865902e+13, 6.662964702658632e-04, 2.216985076700994e-03, -1.554785233828798e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
