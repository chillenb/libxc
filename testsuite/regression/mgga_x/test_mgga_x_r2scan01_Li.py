
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r2scan01_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.037410275989704e+00, -1.413069652815264e+00, -3.262059100180283e-01, -1.840368824037069e-01, -7.184056183384945e-02, -5.937140118485986e-03, -1.442334935024047e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_r2scan01_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.669950890465451e+00, -2.672435437700497e+00, -1.828019626739203e+00, -1.829690476584537e+00, -2.707772842771179e-01, -3.159623131778733e-01, -2.430377220538567e-01, 2.665779724507993e-02, -8.216916974065153e-02, 2.403925514489044e-04, -9.520651081089479e-03, 2.808980546488996e-02, -4.692758577105451e-05, -3.990864257398221e-21]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scan01_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.195112236715514e-04, 0.000000000000000e+00, -2.187262371639808e-04, -8.537311788325634e-04, 0.000000000000000e+00, -8.499932255793128e-04, -2.390296710555416e-01, 0.000000000000000e+00, -1.781161686214922e-01, -3.543751120081068e+00, 0.000000000000000e+00, -9.427722630567966e+02, -9.148525803206061e+01, 0.000000000000000e+00, -6.851520873351276e+05, 2.042608411748392e+01, 0.000000000000000e+00, -8.507493305060365e+02, 3.271620894270357e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scan01_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scan01_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.176621126917781e-02, 1.175606710284072e-02, 1.561880103182449e-02, 1.559289824122725e-02, 6.026355641535924e-02, 4.578131581528594e-02, 1.382440259669196e-01, 1.239905555477703e-02, 2.360303418754523e-01, 2.900026040961655e-04, 9.063789736651498e-10, 1.272706039871861e-02, 1.229947839497027e-18, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
