
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_task_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.297848168786760e-01, -6.292011193377043e-01, -3.334274385880172e-01, -4.560117232761338e-02, -1.960756303520935e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_task_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.730464300380488e-01, -4.233037846379773e-17, -8.661002571817764e-01, -2.394294797436086e-16, -5.471994333751070e-01, 5.853375466495344e-18, -6.900740193392285e-02, -2.268519946863576e-17, -2.614686140708973e-03, -2.236572354516779e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.155451872044479e-07, 0.000000000000000e+00, 0.000000000000000e+00, -1.920082676173540e-02, 0.000000000000000e+00, 0.000000000000000e+00, -3.303290318265267e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.243310492990466e+01, 0.000000000000000e+00, 0.000000000000000e+00, 5.195636932581369e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.311259508227600e-11, 0.000000000000000e+00, 3.471057825249423e-02, 0.000000000000000e+00, 1.233720109547591e-01, 0.000000000000000e+00, 9.683350505160561e-03, 0.000000000000000e+00, 3.016404964779614e-07, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
