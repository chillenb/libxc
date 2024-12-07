
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_bhandhlyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandhlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.110680256778187e-01, -2.900990890269192e-01, -1.807196972201235e-01, -7.351552456105409e-02, -2.974972197376045e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_bhandhlyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandhlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.140759864773457e-01, -2.361713832519181e-01, -3.594280295467689e-01, -2.499095375018285e-01, -2.023282667073683e-01, -1.966669686227086e-01, -5.203628450689889e-02, -3.555089138121427e-02, -7.586470074692439e-03, -2.453539843799461e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_bhandhlyp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandhlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.045149826765442e-02, 2.924820117334720e-02, 2.192774932900085e-02, -1.271344022728777e-02, 4.646917650590301e-02, 3.480665078312204e-02, -8.436825802681701e-02, 3.986771514585075e-01, 2.989990296175629e-01, -5.176003602835856e+00, 1.702787037828197e+01, 1.277088117642037e+01, -2.563429063828265e+04, 7.157809872575426e-18, 5.368349375325281e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
