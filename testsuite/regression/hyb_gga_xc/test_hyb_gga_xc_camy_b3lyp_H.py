
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_camy_b3lyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.475815643519376e-01, -4.111213408482953e-01, -2.353712594281186e-01, -6.622207168582239e-02, -2.112341012090608e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_camy_b3lyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.117526155848954e-01, -2.426334174267675e-01, -5.285939979195401e-01, -2.514212499687130e-01, -2.830566019192489e-01, -1.975342752986723e-01, -5.827976938609369e-02, -4.666762109854466e-02, -5.693635559172133e-03, -3.309251232605039e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_camy_b3lyp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.396521481865559e-02, 2.369104295041123e-02, 1.776147695649069e-02, -1.661253096991197e-02, 3.764003296978145e-02, 2.819338713432885e-02, -9.734375308863147e-02, 3.229284926813911e-01, 2.421892139902259e-01, -3.928748403620982e+00, 1.379257500640840e+01, 1.034441375290050e+01, -1.794400383624976e+04, 5.797825996786096e-18, 4.348362994013478e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
