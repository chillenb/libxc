
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_br89_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.014062987983281e-01, -4.929461477725316e-01, -3.668702125888818e-01, -1.661204655630045e-01, -7.519508807807911e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_br89_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.886005229426255e-01, -1.420092923155131e-16, -7.328742543464650e-01, 9.767168776172069e-17, -4.532965633726714e-01, -2.011795055005906e-16, -1.335012001028787e-01, -9.955034922884368e-17, -3.668259337038568e-02, -2.603588678334526e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.600013120195356e-04, 0.000000000000000e+00, 0.000000000000000e+00, -5.916286811850319e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.856201981867774e-01, 0.000000000000000e+00, 0.000000000000000e+00, -7.077019162278668e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.911577781390648e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.192502080697149e-04, 0.000000000000000e+00, -3.178690224226260e-03, 0.000000000000000e+00, -1.995538958148418e-02, 0.000000000000000e+00, -1.449102730970897e-02, 0.000000000000000e+00, -6.233482688905057e-03, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.816006658230880e-04, 0.000000000000000e+00, 1.017180871752400e-02, 0.000000000000000e+00, 6.385724666074936e-02, 0.000000000000000e+00, 4.637128739106866e-02, 0.000000000000000e+00, 1.994714460449618e-02, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
