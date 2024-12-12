
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_hf_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([6.000231983744743e-01, 1.997172627066726e-01, 3.570570466423461e-02, -9.100753993773489e-02, 1.309500636083732e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_hf_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([1.038639937165935e+00, 1.620000880207321e-16, 5.673098234742757e-01, 6.364528361771169e-17, 2.577267604785272e-01, 1.070757237662185e-18, -1.099550635676783e-01, -3.752998464356106e-17, 1.731129131248706e-02, 8.027138466716062e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_hf_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.905806464636220e-02, 0.000000000000000e+00, 0.000000000000000e+00, 2.425940491637317e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.021575463249492e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.094281009266496e+01, 0.000000000000000e+00, 0.000000000000000e+00, -2.665557952971816e+01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_hf_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.407716881624858e+01, 0.000000000000000e+00, -4.513622077786114e-01, 0.000000000000000e+00, -3.088173737407111e-01, 0.000000000000000e+00, 1.012833817279281e-01, 0.000000000000000e+00, 1.594823451265135e-04, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
