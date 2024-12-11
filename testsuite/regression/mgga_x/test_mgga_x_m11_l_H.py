
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_m11_l_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.641671256714300e-01, -6.399863897121838e-01, -3.315317962684782e-01, -2.405586800618031e-02, -4.088074022309534e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_m11_l_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.170284642198157e-01, -1.100113277606922e-16, -6.585256656151652e-01, -2.901384661902021e-17, -5.076369797817314e-01, 3.778984667742429e-17, -2.756247384644529e-02, -7.741589099525090e-18, -5.428341847414532e-03, -8.857881181338635e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.561946422428966e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.825566248535923e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.909966277641579e-02, 0.000000000000000e+00, 0.000000000000000e+00, -4.039017818302626e+02, 0.000000000000000e+00, 0.000000000000000e+00, -9.461491274616690e+07, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.854083801092623e+00, 0.000000000000000e+00, -2.090881477262768e-01, 0.000000000000000e+00, 7.501443945316828e-02, 0.000000000000000e+00, -4.346347015983779e-03, 0.000000000000000e+00, -1.964380094256747e-05, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
