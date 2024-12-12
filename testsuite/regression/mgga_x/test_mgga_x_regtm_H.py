
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_regtm_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.919843978961326e-01, -6.235841247247478e-01, -3.728441343251529e-01, -1.216520988216178e-01, -1.655470689618618e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_regtm_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.223491018283836e-01, -6.190086868579379e-17, -8.053648633009493e-01, -2.349561356316372e-16, -4.477235636613823e-01, 6.785739841582763e-20, -1.057060997156997e-01, -6.966980511095977e-17, -8.513365497426652e-03, -1.400544465979515e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.237892759194305e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.134125421366036e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.644061140052775e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.034383372538874e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.739833484857130e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.986758047492939e-02, 0.000000000000000e+00, 2.538238118849552e-02, 0.000000000000000e+00, 3.110805181830354e-02, 0.000000000000000e+00, 4.177341699339111e-02, 0.000000000000000e+00, 7.194259459298397e-03, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
