
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mb3lyp_rc04_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.119055981203096e-01, -4.759892979581590e-01, -2.969802079751910e-01, -1.171744399151011e-01, -4.341806559164973e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mb3lyp_rc04_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.785657536579330e-01, -8.832608842859714e+02, -5.924848570644260e-01, -7.599378562874307e+02, -3.386684583585245e-01, -3.469602595635906e+02, -8.973789351488634e-02, -3.778263439568536e+01, -1.169487812750298e-02, -1.042524766019141e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mb3lyp_rc04_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.505015750542190e-02, 2.369104295041123e-02, 1.776147695649069e-02, -1.830735392729338e-02, 3.764003296978145e-02, 2.819338713432885e-02, -1.214902915585818e-01, 3.229284926813911e-01, 2.421892139902259e-01, -7.453445188105283e+00, 1.379257500640840e+01, 1.034441375290050e+01, -3.691337851912702e+04, 5.797825996786096e-18, 4.348362994013478e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
