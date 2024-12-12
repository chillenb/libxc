
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rppscan_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.600108934241007e-13, -2.107855556765514e-12, -2.909329027689367e-12, -1.332128851672110e-13, -1.165557644703438e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rppscan_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.035090728203278e-04, -2.199798179630562e-01, -3.943870693179131e-03, -2.101319763062403e-01, -8.144130768021729e-03, -1.813309671284328e-01, -3.842121760199537e-02, -1.091393373285479e-01, -9.724226768849846e-02, -9.891645420874035e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rppscan_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.233851539257247e-03, 8.467703078514494e-03, 4.233851539257247e-03, 4.885035171849409e-03, 9.770070343698816e-03, 4.885035171849409e-03, 4.742997516497319e-02, 9.485995032994637e-02, 4.742997516497319e-02, 1.153250791065013e+01, 2.306501582130025e+01, 1.153250791065013e+01, 2.072111432257701e+05, 4.144222864515402e+05, 2.072111432257701e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rppscan_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.009767073754148e-02, -9.979290553912836e-03, -8.398788787689286e-03, -8.261879068778776e-03, -1.631690760221510e-02, -1.625762109617745e-02, -7.556532297031493e-02, -7.556336542252599e-02, -1.419598218080804e-01, -1.419598229148761e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
