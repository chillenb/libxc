
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optb86b_vdw_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb86b_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218547681003738e-01, -5.690334285660764e-01, -3.464766382551968e-01, -1.353500815996594e-01, -1.868886938930327e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optb86b_vdw_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb86b_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285212367564024e-01, -1.343115308820988e-16, -7.281419977146988e-01, -1.777963844989584e-16, -4.100897997467734e-01, 3.621187761245933e-17, -1.017218883392981e-01, -8.123318290872027e-17, -1.708438011260676e-02, -2.894518409010876e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optb86b_vdw_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb86b_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.486279317472725e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.419910406574138e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.133004022848929e-01, 0.000000000000000e+00, 0.000000000000000e+00, -8.863520831862335e+00, 0.000000000000000e+00, 0.000000000000000e+00, -6.260068909577848e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
