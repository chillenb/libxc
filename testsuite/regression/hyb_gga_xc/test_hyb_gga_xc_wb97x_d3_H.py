
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wb97x_d3_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.236392381495038e-01, -3.672522981664084e-01, -1.870313280147896e-01, -4.179945765699361e-02, -9.328234378597368e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wb97x_d3_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.004041742011956e-01, -2.600793446110508e-01, -5.198029324718435e-01, -2.640263827656717e-01, -2.305476610375357e-01, -2.158093031873883e-01, 2.320097737446522e-03, 3.199280336216548e-02, -1.201765011596979e-03, 2.283049110477376e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wb97x_d3_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.482228311441493e-01, 0.000000000000000e+00, -3.913186273195054e+20, 2.988956838852418e-05, 0.000000000000000e+00, -2.211153507105743e+20, -1.030868121740110e-01, 0.000000000000000e+00, -2.584692574899597e+18, -8.399228199785060e+00, 0.000000000000000e+00, 4.706577382878476e+19, -5.017411643289645e-01, 0.000000000000000e+00, 7.001701452304167e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
