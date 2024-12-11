
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_21_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.004493756175965e+00, -1.311428665020233e+00, -2.536044427317555e-01, -1.854451143704152e-01, -5.425554821243817e-02, -1.025696673693408e-02, -1.924832872574300e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_21_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.872304649834228e+00, -2.875103370794237e+00, -1.992382796003604e+00, -1.994178121848908e+00, -3.271836625323150e-01, -3.270477224815693e-01, -2.587876752586725e-01, -1.273674563723997e-02, -7.475360754043044e-02, -4.038945690204681e-04, -1.339172648321994e-02, -1.329625623726172e-02, -2.697782746900265e-04, -2.014389627049625e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_21_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.949544078831368e-04, 0.000000000000000e+00, -7.922956791606251e-04, -2.869112680822788e-03, 0.000000000000000e+00, -2.861970648823951e-03, -2.509866220727529e-02, 0.000000000000000e+00, -2.685085650745006e-02, -1.262709717517159e+01, 0.000000000000000e+00, -8.160205741424193e+00, -6.167934280613951e+01, 0.000000000000000e+00, -2.034522045771966e+04, -1.510690942162796e-01, 0.000000000000000e+00, -7.298238267228226e+00, -3.080811629123331e-01, 0.000000000000000e+00, 2.498987250035347e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_21_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.865650400866680e-02, 2.862749601495409e-02, 3.578765620389456e-02, 3.578960826694177e-02, -2.022819153633950e-03, -2.032959279355289e-03, 3.257344640531477e-01, 1.046642698547707e-04, 2.020283791553338e-02, 8.289453790460463e-06, 2.243772936660718e-06, 1.065196639178763e-04, 3.740599921842857e-11, -8.637087942791851e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
