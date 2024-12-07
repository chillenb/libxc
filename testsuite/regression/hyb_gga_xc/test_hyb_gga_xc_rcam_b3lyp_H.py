
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_rcam_b3lyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.498737791733125e-01, -3.117771857701879e-01, -1.369834515547962e-01, 1.332365692649803e-03, 4.021250929556954e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_rcam_b3lyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.121881651730400e-01, -2.361713832519182e-01, -4.334009924492609e-01, -2.499095375018285e-01, -1.973186323296454e-01, -1.966669686227087e-01, -3.442226657201769e-03, -3.555089138121423e-02, 7.015585115749496e-04, -2.453539843799460e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_rcam_b3lyp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.225076447693871e-02, 2.924820117334720e-02, 2.192774932900085e-02, -1.385834997094206e-02, 4.646917650590301e-02, 3.480665078312204e-02, -5.630404750442328e-02, 3.986771514585075e-01, 2.989990296175629e-01, -9.703677366152394e-02, 1.702787037828197e+01, 1.277088117642037e+01, -1.327856538379853e+02, 7.157809872575426e-18, 5.368349375325281e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
