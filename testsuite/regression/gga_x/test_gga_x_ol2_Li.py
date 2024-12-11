
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ol2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.803992228914872e+00, -1.299588628909210e+00, -4.208489356101423e-01, -1.590036058386003e-01, -7.858511914804887e-02, -2.503755077139868e+00, -5.928644036796430e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ol2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.146964632420220e+00, -2.148625632153407e+00, -1.546452383959297e+00, -1.547705623681984e+00, -2.814042729813179e-01, -2.807898671705698e-01, -1.856220261734524e-01, 3.432635299247994e+00, -7.427271062573473e-02, 4.304893762291093e+01, 3.235938727406444e+00, 3.374239598187032e+00, 7.553925575163595e+01, 8.881367298821556e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ol2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.410659926825840e-04, 0.000000000000000e+00, -4.403478846868381e-04, -9.780583049079760e-04, 0.000000000000000e+00, -9.756380735862972e-04, -1.356134732015166e-01, 0.000000000000000e+00, -1.358241774268734e-01, -1.231074232566370e+01, 0.000000000000000e+00, -6.624970543760395e+04, -6.674138921274188e+01, 0.000000000000000e+00, -6.547916630895298e+10, -5.417945626239153e+04, 0.000000000000000e+00, -5.578473689205866e+04, -3.289627231292270e+11, 0.000000000000000e+00, -1.287927828567658e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
