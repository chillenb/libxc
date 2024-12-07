
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_15_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.106886738679211e+00, -1.474772352667421e+00, -3.225978866102023e-01, -1.890192676400360e-01, -7.222264207502294e-02, -1.218933527023757e-02, -2.277748771546983e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_15_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.609502428008977e+00, -2.612051480316367e+00, -1.740903013419483e+00, -1.742345397073799e+00, -3.997100799941991e-01, -4.006667228921256e-01, -2.414913364301262e-01, -1.487434568138111e-02, -8.729391429466733e-02, -4.717454193858021e-04, -1.631887959482732e-02, -1.552762526373434e-02, -3.288568245859459e-04, -2.240064899990315e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_15_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.209673239782041e-04, 0.000000000000000e+00, -7.185312532688822e-04, -2.822096995017226e-03, 0.000000000000000e+00, -2.814578783983374e-03, -6.415890238475855e-02, 0.000000000000000e+00, -6.784078423298581e-02, -1.104174605259358e+01, 0.000000000000000e+00, -1.651191071355251e+01, -9.448641052186625e+01, 0.000000000000000e+00, -4.131917817113638e+04, 3.234606962251675e-01, 0.000000000000000e+00, -1.476441746897872e+01, 6.910909084396651e-01, 0.000000000000000e+00, -1.870580009599012e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_15_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_15_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.161489761850900e-02, 3.158777385716221e-02, 4.549521603088096e-02, 4.548513588118042e-02, 1.869739298545988e-02, 2.041141194702539e-02, 3.436528495773474e-01, 2.111359480655077e-04, 2.861576332909496e-01, 1.683494707321548e-05, -9.851123950827542e-08, 2.147901396954057e-04, -6.644273362879894e-16, 8.160071795949438e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
