
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_q2d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218547994840735e-01, -5.688848986480545e-01, -3.452693575690670e-01, -3.472194892461343e-02, -2.577068273456544e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_q2d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285211952012457e-01, 7.642394169951012e-17, -7.288561787016720e-01, -1.917702742272153e-16, -4.161383842446514e-01, -1.139897435588458e-17, -1.636326445282658e-01, -2.255255878932367e-17, -5.123789110582278e-04, -5.452458362609565e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_q2d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.487558566196737e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.377538666175141e-02, 0.000000000000000e+00, 0.000000000000000e+00, -9.657520102671770e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.320741491505124e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.348602762287991e+02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
