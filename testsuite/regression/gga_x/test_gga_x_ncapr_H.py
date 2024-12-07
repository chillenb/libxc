
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ncapr_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncapr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220388454037054e-01, -5.785930139222059e-01, -3.619231346116032e-01, -1.544224672108620e-01, -2.569914101957279e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ncapr_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncapr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282716204259889e-01, -1.140840624237372e-16, -7.164611146315840e-01, -2.472860699759451e-16, -3.972561192688094e-01, 8.310226242875294e-18, -9.304618111072141e-02, -7.322348053570146e-17, 1.145883089519429e-01, -1.564291894436816e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ncapr_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncapr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.707974331779951e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.554519793775861e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.863071375735916e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.270244048347244e+01, 0.000000000000000e+00, 0.000000000000000e+00, -3.653733479709051e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
