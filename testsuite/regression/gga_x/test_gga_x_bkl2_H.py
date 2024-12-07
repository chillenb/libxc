
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bkl2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218168328020227e-01, -5.671738330670448e-01, -3.433716985894962e-01, -1.292110834002722e-01, -4.101570869491527e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bkl2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285717414024626e-01, -6.776342009799131e-17, -7.304157433254403e-01, -1.510578886111978e-16, -4.132286689895159e-01, -1.584231103231453e-17, -1.100840809755518e-01, -3.228788481752305e-17, -5.469096828656643e-03, -2.382241534312850e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bkl2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.935766437754375e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.199128765499300e-02, 0.000000000000000e+00, 0.000000000000000e+00, -9.740401098983405e-02, 0.000000000000000e+00, 0.000000000000000e+00, -7.000932348741967e+00, 0.000000000000000e+00, 0.000000000000000e+00, 2.681750289704260e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
