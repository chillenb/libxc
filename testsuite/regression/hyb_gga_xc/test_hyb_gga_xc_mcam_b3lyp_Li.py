
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mcam_b3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.486755092383820e+00, -1.064705310700007e+00, -3.010183351310098e-01, -1.112915541563504e-01, -5.323193131272991e-02, -8.556693419212499e-02, -3.329596171563445e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mcam_b3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.855949935025587e+00, -1.857495559410204e+00, -1.275661889483846e+00, -1.276596267782830e+00, -3.522816552294315e-01, -3.525426124525230e-01, -1.455119682336123e-01, -1.133222025803346e-01, -4.943795304543423e-02, -4.346462108217933e-02, -2.654327424993492e-02, -2.674394598866954e-02, -4.679182461900143e-03, -4.130563549000532e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mcam_b3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.084766969443428e-04, 4.230480491699886e-06, -2.078770482086570e-04, -7.584947064375717e-04, 2.954022849291356e-05, -7.565819635994704e-04, -4.827668043446415e-02, 3.866747919504811e-02, -4.811893706936360e-02, -2.818025986796687e+00, 3.722869163256963e+00, -8.277523622182039e+02, -4.753742963896426e+01, 1.909121184807025e+01, -3.007186615567474e+07, -7.221843916508915e+02, 6.428238830639903e-02, -7.233367255061581e+02, -8.928013866737956e+07, 0.000000000000000e+00, -2.659558958773407e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
