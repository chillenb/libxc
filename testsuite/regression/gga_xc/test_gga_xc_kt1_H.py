
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_kt1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.545186491584373e-01, -6.096701891162648e-01, -3.600901133244773e-01, -1.005354401395230e-01, -5.661926116267190e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_kt1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.647736121436500e-01, -2.701820894353360e-01, -7.622958456425593e-01, -2.578659189064837e-01, -4.617713289771110e-01, -2.012317406014648e-01, -1.317853957389462e-01, -9.405999515663745e-02, -7.462568705062886e-03, -6.957283995407767e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_kt1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.005640300156342e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.623184017064704e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.214406033937811e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.995405584225613e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.999999977351767e-02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
