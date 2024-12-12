
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_jk_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.216255246370611e-01, -5.679043028668148e-01, -3.645319852773768e-01, -1.381529140803655e-01, -4.898572739022287e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_jk_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.288243881681026e-01, -1.079374559955548e-16, -7.358776232348644e-01, -2.825401713810673e-16, -4.439588346388636e-01, -3.001544078434344e-17, -1.709185837367708e-01, -6.389410667288574e-17, -7.714345826596777e-02, -7.462911514742288e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.180530625285134e-04, 0.000000000000000e+00, 0.000000000000000e+00, -1.585024024316356e-02, 0.000000000000000e+00, 0.000000000000000e+00, -8.904128093183418e-02, 0.000000000000000e+00, 0.000000000000000e+00, 5.948068149097324e+00, 0.000000000000000e+00, 0.000000000000000e+00, 9.354042459372322e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-2.717062949698101e-08, 0.000000000000000e+00, -5.059645105564334e-04, 0.000000000000000e+00, -6.563656206758626e-03, 0.000000000000000e+00, -1.487652375697993e-02, 0.000000000000000e+00, -1.314150295360382e-02, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
