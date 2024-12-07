
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_revtpssh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.848250387128550e+00, -1.287856258035703e+00, -3.461591105734357e-01, -1.637236085889078e-01, -6.696255955043325e-02, -1.848373318387932e-02, -3.454727313937018e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_revtpssh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.237674062380352e+00, -2.239608294957965e+00, -1.598955942203032e+00, -1.600534796869071e+00, -3.300193077399844e-01, -3.312298120650285e-01, -2.196355895651931e-01, -4.638146928362256e-01, -7.420764864159136e-02, -1.588630525577807e-02, -2.468567544534154e-02, -2.450952834819904e-02, -4.987395204095092e-04, -3.545585856777953e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.129400919658022e-03, 1.895928374572110e-04, -1.127885169403036e-03, -2.009345379126945e-03, 2.954338226692877e-04, -2.005290850960016e-03, -5.417778496335106e-02, -3.100189249575413e-02, -5.338276108240576e-02, 1.699053803183887e+00, 8.734989155463344e+01, 8.277457868796283e+02, -2.938807510313253e+01, 4.833712340946774e+01, 9.921882531834922e+04, -3.968579298679128e-01, -7.009599139035285e-04, -3.704963473580382e-01, -1.824905048737550e+00, 1.011868011772730e-05, -2.612166441249800e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_revtpssh_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_revtpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [5.591399617696770e-02, 5.604018848122434e-02, 3.633775196441673e-02, 3.642753294372224e-02, 2.568915067884255e-03, 2.531999555747004e-03, -9.455001597066026e-03, -1.169741281591261e+00, -1.377955917719306e-02, -5.783869036481568e-02, 8.468777522770021e-14, 1.220041893175035e-10, -2.845928495734609e-31, 9.477408664191114e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
