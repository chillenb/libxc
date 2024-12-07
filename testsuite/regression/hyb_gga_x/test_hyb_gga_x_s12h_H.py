
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_s12h_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.780779740205662e-01, -4.329870501782291e-01, -2.716814072265023e-01, -1.021112677996870e-01, -5.403513206178220e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_s12h_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.374342658774703e-01, 1.838627314781663e-17, -5.511566068818278e-01, -2.112706643171393e-16, -2.850430437733381e-01, -6.614738817166316e-17, -1.111361214983466e-01, -5.473049856205730e-17, -7.201169797116053e-03, -1.378778090403201e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_s12h_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.651945581375639e-05, 0.000000000000000e+00, 0.000000000000000e+00, -1.215080553607509e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.685971229034138e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.815376071986684e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.808409917874974e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
