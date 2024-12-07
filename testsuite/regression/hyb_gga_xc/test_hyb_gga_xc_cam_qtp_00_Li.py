
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_qtp_00_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.304029784722133e-01, -5.865047083546117e-01, -1.135090604149705e-01, -3.692338299420490e-02, -1.139812646520108e-02, -1.502275182009831e-02, -4.881874742923615e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_qtp_00_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.048795553674048e+00, -1.049595965311358e+00, -7.211213316460348e-01, -7.215766072796604e-01, -2.209317846781798e-01, -2.212452446117201e-01, -5.454801561828541e-02, -9.481772240085720e-02, -1.282557224333306e-02, -3.981161525278803e-02, -7.184692006759276e-03, -7.291747164120206e-03, -7.264333058266166e-04, -7.123181193843216e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_qtp_00_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_00", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.068064875442459e-04, 4.178252337481369e-06, -1.065473139933022e-04, -3.666178588271739e-04, 2.917553431398870e-05, -3.659531391500172e-04, 5.017669262007507e-03, 3.819010290868950e-02, 5.165330758124745e-03, -6.185252605321838e-01, 3.676907815562432e+00, -1.178042782440993e+02, -7.045765130382654e+00, 1.885551787463729e+01, -4.365258830404668e+06, -1.048053662169847e+02, 6.348877857422126e-02, -1.049724947737005e+02, -1.296002012913574e+07, 0.000000000000000e+00, -3.860650101445267e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
