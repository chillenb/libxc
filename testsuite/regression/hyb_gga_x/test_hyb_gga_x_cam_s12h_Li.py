
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_cam_s12h_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.313875135183692e+00, -9.323075238222575e-01, -2.890355178930181e-01, -1.057717925027084e-01, -5.401403501763136e-02, -1.283043470253068e-02, -2.396510503364543e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_cam_s12h_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.682486257638381e+00, -1.684249207722740e+00, -1.075222971740524e+00, -1.076317238680424e+00, -3.026777015460395e-01, -3.028131240053553e-01, -1.406607840178173e-01, -1.631654268182828e-02, -5.153587384090982e-02, -5.179644982388686e-04, -1.715610513858277e-02, -1.703225712212346e-02, -3.459710859395453e-04, -2.459539258729600e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_cam_s12h_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.333250407517464e-04, 0.000000000000000e+00, -1.325883600369930e-04, -9.296511238665532e-04, 0.000000000000000e+00, -9.262202106182072e-04, -4.277070527434511e-02, 0.000000000000000e+00, -4.262934809556434e-02, -8.993955667761234e-01, 0.000000000000000e+00, -1.118004126051641e-01, -4.511893796571704e+01, 0.000000000000000e+00, -7.145686734004846e-01, -1.136236096903394e-01, 0.000000000000000e+00, -1.060998390846235e-01, -5.201789852399687e-01, 0.000000000000000e+00, -7.445824680423193e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
