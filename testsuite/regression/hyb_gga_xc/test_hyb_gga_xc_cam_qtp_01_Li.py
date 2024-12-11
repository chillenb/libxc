
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_qtp_01_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.313773358161396e+00, -9.091488035202721e-01, -1.485156713616605e-01, -3.988904072439973e-02, -5.673632772722966e-03, -3.044661346357897e-03, -5.661550944004586e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_qtp_01_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.677432245528072e+00, -1.678879925136862e+00, -1.132033813389754e+00, -1.132900978414181e+00, -2.881002195828723e-01, -2.883975563572588e-01, -6.635737673983128e-02, -9.161197900361837e-02, -9.291464783834259e-03, -3.911643297009667e-02, -3.905870546926224e-03, -3.996623687591323e-03, -5.492216192519971e-05, -1.315393483774537e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_qtp_01_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.747708468789581e-04, 4.178252337481369e-06, -1.743017573591559e-04, -5.950648095751225e-04, 2.917553431398870e-05, -5.937584558359181e-04, 4.386745832092818e-03, 3.819010290868950e-02, 4.549429762507082e-03, -3.957089623601712e-01, 3.676907815562432e+00, 2.758783208468010e+00, -2.476708454400660e-01, 1.885551787463729e+01, 1.414163995060173e+01, 3.263191418164080e-02, 6.348877857422126e-02, 3.280394608419628e-02, -3.023691745456475e-09, 0.000000000000000e+00, -1.462750290504580e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
