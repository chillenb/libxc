
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_pbe1w_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbe1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.863224178458176e+00, -1.337654318927865e+00, -4.312499969751179e-01, -1.759402882359517e-01, -8.435386842205386e-02, -2.231119735219223e-02, -4.262194370385825e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_pbe1w_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbe1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.355288202825093e+00, -2.357286639776060e+00, -1.618270980864968e+00, -1.619534067213533e+00, -4.273294706378864e-01, -4.275214154239498e-01, -2.286849831641767e-01, -1.318950440313553e-01, -8.467997551563773e-02, 2.335161899354947e-01, -2.967927659312455e-02, -2.950655466768795e-02, -6.039251058680320e-04, -4.676024465711050e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_pbe1w_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_pbe1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.211444407056732e-04, 6.801319058524462e-05, -2.202656020840666e-04, -9.001790695927849e-04, 2.205935195019102e-04, -8.969368425765038e-04, -7.235925518765982e-02, 4.624962008092946e-03, -7.217495898903827e-02, -1.448102890734826e+00, 5.004078999583691e+00, 2.224322990989513e+00, -5.933950007487177e+01, 1.671437152402882e+01, 6.580733081577065e+00, -2.820949115087826e-01, 2.484309204426431e-04, -2.634186767075343e-01, -1.293190869033318e+00, 2.377535476784046e-06, -1.851070193021428e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
