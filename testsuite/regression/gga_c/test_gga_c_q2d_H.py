
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_q2d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.206369589024619e-02, -1.817875322923838e-02, -8.087380875854370e-03, -1.590585980923498e-04, -4.459907078386793e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_q2d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.697773492473688e-02, 1.374344947415270e-01, -4.114828852223622e-02, 1.571204532976518e+01, -2.759482338526383e-02, 8.789797976867510e+00, -9.401020261615542e-04, 6.982805656623159e-02, 2.050482388798646e-05, 2.950551024260514e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_q2d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.644029075325091e-02, 3.288058150650183e-02, 1.644029075325091e-02, 1.020691046018351e-02, 2.041382092036701e-02, 1.020691046018351e-02, 4.208657284117645e-02, 8.417314568235290e-02, 4.208657284117645e-02, 8.772533701053922e-02, 1.754506740210784e-01, 8.772533701053922e-02, -7.655373009304505e+01, -1.531074601860901e+02, -7.655373009304505e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
