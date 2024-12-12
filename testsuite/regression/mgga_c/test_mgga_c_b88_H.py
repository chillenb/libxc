
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_b88_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.384472456262437e-13, -2.203701676714032e-12, -2.744185495231219e-12, -1.987992135596366e-14, -5.025594315772112e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_b88_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.972193593163745e-05, -1.167394320204798e-02, -4.124232085930518e-03, -1.022599906912655e-02, -7.688228216478149e-03, -7.293568523497006e-03, -1.100331700995702e-02, -1.628343498606153e-03, -1.817875316141456e-02, -2.625344534596079e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b88_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.669913624141088e-03, 0.000000000000000e+00, 0.000000000000000e+00, 5.108437969902528e-03, 0.000000000000000e+00, 0.000000000000000e+00, 4.477487944680474e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.302754269139653e+00, 0.000000000000000e+00, 0.000000000000000e+00, 3.873665573542879e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b88_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-8.752687492381403e-03, 0.000000000000000e+00, -8.782882832253354e-03, 0.000000000000000e+00, -1.540349891106918e-02, 0.000000000000000e+00, -2.164088635121681e-02, 0.000000000000000e+00, -2.653838335689744e-02, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
