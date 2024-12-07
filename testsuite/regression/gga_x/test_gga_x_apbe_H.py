
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_apbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.221110169942904e-01, -5.810260404602611e-01, -3.646851384512016e-01, -1.369035771693801e-01, -7.397020325420629e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_apbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.281806587679204e-01, -4.769609964338315e-17, -7.149622972314774e-01, -1.281254872169637e-16, -3.983249343768476e-01, -1.145676826216601e-17, -1.442644835749359e-01, -9.162701224363326e-17, -9.856842828624209e-03, -2.914653868474883e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_apbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.995100088117784e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.774820525584861e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.920156269532033e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.308076809943576e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.675449692752607e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
