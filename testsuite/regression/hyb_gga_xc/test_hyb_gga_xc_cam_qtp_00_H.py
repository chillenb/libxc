
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_qtp_00_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.377428220229668e-01, -2.156794195352217e-01, -1.137680883393392e-01, -2.084734599190195e-02, -5.667609333209677e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_qtp_00_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.315975511083695e-01, -2.429735244886017e-01, -2.857896348887277e-01, -2.515008137827595e-01, -1.472044840998330e-01, -1.975799230184599e-01, -2.163077842644789e-02, -4.725271213629887e-02, -1.765512371643035e-03, -3.354288674121121e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_qtp_00_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.942291104268552e-03, 2.339856093867776e-02, 1.754219946320068e-02, -8.069440734068673e-03, 3.717534120472241e-02, 2.784532062649763e-02, -4.030491981306455e-02, 3.189417211668060e-01, 2.391992236940503e-01, -9.750543694567424e-01, 1.362229630262558e+01, 1.021670494113630e+01, -4.614172333396486e+03, 5.726247898060341e-18, 4.294679500260225e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
