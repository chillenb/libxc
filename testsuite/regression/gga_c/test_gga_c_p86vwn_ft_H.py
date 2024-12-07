
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_p86vwn_ft_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.202368366850009e-02, -1.834278646004013e-02, -6.488505678974788e-03, 7.767047748883812e-03, -1.560270763521161e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_p86vwn_ft_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.685274334945465e-02, -2.703381432200373e-01, -4.454119273613588e-02, -2.567421119698116e-01, -3.799993252170022e-02, -1.950387240452968e-01, 2.830156652266439e-03, -5.799411406289971e-02, -1.992584993558445e-03, -6.955993051274208e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_p86vwn_ft_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.508842424107408e-02, 3.017684848214816e-02, 1.508842424107408e-02, 1.196630080563921e-02, 2.393260161127842e-02, 1.196630080563921e-02, 7.181846072340548e-02, 1.436369214468110e-01, 7.181846072340548e-02, 1.059694824503271e+00, 2.119389649006542e+00, 1.059694824503271e+00, -1.061495594261821e+00, -2.122991188523642e+00, -1.061495594261821e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
