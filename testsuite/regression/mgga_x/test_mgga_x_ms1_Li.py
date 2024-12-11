
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.907442721398207e+00, -1.251652127900726e+00, -2.551759006957331e-01, -1.749553347874425e-01, -5.630709847067722e-02, -1.599607658893805e-02, -2.758346869253733e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.688357841228442e+00, -2.690832584923555e+00, -1.894536356980129e+00, -1.896637167863604e+00, -3.468687384335194e-01, -3.471804210614700e-01, -2.386421590694511e-01, -2.033940779624496e-02, -8.097127699002457e-02, -6.456880840515737e-04, -2.140827458190753e-02, -2.123146167292991e-02, -4.312835992804673e-04, -1.910294188008724e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.937701137224461e-04, 0.000000000000000e+00, -4.917489266039556e-04, -2.472508854196534e-03, 0.000000000000000e+00, -2.468367412066051e-03, -3.069289227109531e-01, 0.000000000000000e+00, -3.075384063182969e-01, -5.137585258084326e+00, 0.000000000000000e+00, -1.245901098316482e-01, -1.528055229376174e+02, 0.000000000000000e+00, -7.975402546178242e-01, -5.312417540225655e-05, 0.000000000000000e+00, -1.182261644172310e-01, 2.592323196615042e-09, 0.000000000000000e+00, -3.001459541110490e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms1_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.085844316000916e-02, 2.082793904627710e-02, 3.312751725031543e-02, 3.318722781648745e-02, 1.204081253380386e-03, 1.295201968595891e-03, 1.507118969851819e-01, 7.796729907858791e-18, 4.935976507250638e-02, 1.131144506043248e-18, -1.274206886693469e-20, 2.778204598835961e-18, -3.191695318823092e-19, 1.485305714382870e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
