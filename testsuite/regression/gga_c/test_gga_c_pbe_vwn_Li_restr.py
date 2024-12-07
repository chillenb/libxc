
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_vwn_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.050880528759774e-02, -4.393154886530265e-02, -3.098912730719098e-03, -2.942622588974165e-02, -3.698810668355166e-03, -7.667828554293599e-09, -1.830462919677139e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_vwn_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.165194096362083e-01, -1.020378881164970e-01, -1.604446331565089e-02, -4.350259355169452e-02, -1.657012740695420e-02, -4.976076056158009e-08, -1.240702614757986e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_vwn_Li_restr_1_vsigma():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.598660050654304e-05, 1.493543891292514e-04, 3.142806004396667e-03, 4.569106567400131e+00, 2.736222093779062e+01, 1.691407066622557e-04, 1.677067683822522e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
