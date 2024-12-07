
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2bs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.937538147767521e+00, -1.360798914312273e+00, -3.825522888698923e-01, -1.742296776375720e-01, -7.700983555477468e-02, -1.851859887871966e-02, -3.460473058709119e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2bs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.521614796461174e+00, -2.524026380902057e+00, -1.732724779988404e+00, -1.734664188191528e+00, -4.067533293723452e-01, -4.116997765243947e-01, -2.288065182800687e-01, -2.353561213485589e-02, -8.604329290081449e-02, -7.479200723727097e-04, -2.474363248590829e-02, -2.456629348724751e-02, -4.995692079033905e-04, -3.551483345001878e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2bs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.041562553927157e-04, 0.000000000000000e+00, -1.033818026053854e-04, -4.300856400957488e-04, 0.000000000000000e+00, -4.226997623437234e-04, -5.397008963099725e-02, 0.000000000000000e+00, -4.764938011509604e-02, -2.021651181719537e+00, 0.000000000000000e+00, -2.991548054022322e-01, -3.640458139002768e+01, 0.000000000000000e+00, -1.916698202894489e+00, -3.041976107757595e-01, 0.000000000000000e+00, -2.838585809893152e-01, -1.395290207910287e+00, 0.000000000000000e+00, -1.997214198878529e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2bs_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2bs_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2bs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.132589901336758e-05, 2.069381125102352e-11, 1.040544564141041e-04, 2.082206593498560e-17, 1.449434171276972e-03, 4.714839704386802e-12, 1.666095812107801e-02, 2.098042097048271e-24, 1.325900747577418e-07, 4.305124186499053e-22, 1.466842310853693e-13, 8.425806116442476e-25, 6.928062081497584e-28, 9.318134492842702e-23]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
