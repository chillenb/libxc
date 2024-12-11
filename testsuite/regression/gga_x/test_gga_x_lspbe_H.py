
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lspbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.220350471457701e-01, -5.774866795156860e-01, -3.593209989411344e-01, -1.328377815105750e-01, -6.201994249629700e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lspbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.282816086378144e-01, -9.059719239437029e-17, -7.188202432381732e-01, -2.139509595972470e-16, -4.018595277888787e-01, -1.549615427693429e-17, -1.410107866661850e-01, -7.350157721499236e-17, -1.590211955730090e-04, -4.620430790172784e-22])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lspbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.684885672315730e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.376423156707630e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.686764104752111e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.064118422342911e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.204625373296191e+02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
