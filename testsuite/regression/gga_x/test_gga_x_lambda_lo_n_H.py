
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lambda_lo_n_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220350553355097e-01, -5.775089515796760e-01, -3.594309046980264e-01, -1.343932629822215e-01, -7.396948623153268e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lambda_lo_n_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282815759061094e-01, -4.285894482846990e-17, -7.187351604148711e-01, -2.352788126891114e-16, -4.014751360985652e-01, -9.870246579613110e-18, -1.383100776637808e-01, -8.199602960362586e-17, -9.855668420497982e-03, -5.145866048149639e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lambda_lo_n_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.684952627811904e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.381754521334849e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.698359322278006e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.601556664802894e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.537499364376387e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
