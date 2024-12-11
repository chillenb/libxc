
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_p86_ft_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.217521646876693e-02, -1.852706915030256e-02, -6.729525474225508e-03, 7.725196263655024e-03, -1.551468681653548e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_p86_ft_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.692949770222917e-02, -2.218122190599615e-01, -4.462536252717236e-02, -2.122216756948365e-01, -3.825623333227448e-02, -1.661917628949329e-01, 2.741903671040950e-03, -4.897082562749759e-02, -1.979020007809162e-03, -7.162845481729079e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_p86_ft_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.508842424107408e-02, 3.017684848214816e-02, 1.508842424107408e-02, 1.196630080563921e-02, 2.393260161127842e-02, 1.196630080563921e-02, 7.181846072340548e-02, 1.436369214468110e-01, 7.181846072340548e-02, 1.059694824503271e+00, 2.119389649006542e+00, 1.059694824503271e+00, -1.061495594261821e+00, -2.122991188523642e+00, -1.061495594261821e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
