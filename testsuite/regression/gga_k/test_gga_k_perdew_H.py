
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_perdew_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.035145926362095e+00, 1.687563385201089e+00, 6.150056313329013e-01, 9.650725555212467e-02, 7.635308941394778e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_perdew_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.388870145984856e+00, 5.056115044449710e-17, 2.673192214651205e+00, 8.022820791416932e-16, 8.768229946279942e-01, 1.315130475203399e-16, 9.889426659763403e-03, 7.450690352594751e-18, -7.611696146902004e-02, -7.404784274235774e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_perdew_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.662551777073670e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.475613194618170e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.236292356424785e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.699160025458832e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.625102380014253e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
