
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3lyp_mcm2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.035345463690175e-01, -4.678607046328017e-01, -2.899847452879044e-01, -1.145333306890661e-01, -4.379200433471690e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3lyp_mcm2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.701681256597491e-01, -2.265353148631948e-01, -5.836978122701152e-01, -2.393149928111334e-01, -3.301492621232678e-01, -1.883970147790243e-01, -8.557081869266399e-02, -3.510285795822587e-02, -1.160692055226621e-02, -2.458788696928074e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3lyp_mcm2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.523828447423976e-02, 2.755473032541040e-02, 2.065813264285170e-02, -1.853619585138475e-02, 4.377861118621124e-02, 3.279134570277928e-02, -1.230089202030708e-01, 3.755937443890600e-01, 2.816869858027060e-01, -7.546613252952407e+00, 1.604195668337945e+01, 1.203144715630564e+01, -3.737479575061611e+04, 6.743372680953310e-18, 5.057521946493947e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
