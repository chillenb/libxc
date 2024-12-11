
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_o3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_o3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.182659140819324e+00, -2.076039074122467e+00, -2.057032407223399e-01, -5.190720251485033e-02, -5.748993546023952e-03, -2.997779326610814e-03, -5.528662946219075e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_o3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_o3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.267898731555880e+00, -4.271602014399217e+00, -2.891645668618522e+00, -2.894264702067284e+00, -4.489337302674207e-01, -4.489967438501393e-01, -9.354437038669476e-02, -9.114532633810751e-02, -9.723347960625130e-03, -3.867558461701569e-02, -3.847852560892251e-03, -3.938559412762951e-03, -5.321592118148884e-05, -1.296428093551048e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_o3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_o3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.423195986397949e-04, 4.230480491699886e-06, -2.424719635891028e-04, -2.038063256851223e-04, 2.954022849291356e-05, -2.041647318972207e-04, 2.795169083699712e-02, 3.866747919504811e-02, 2.808056040099818e-02, -3.806351276936242e-01, 3.722869163256963e+00, 2.793323514115996e+00, -1.063710591794866e-03, 1.909121184807025e+01, 1.431841046228609e+01, 3.310416086969142e-02, 6.428238830639903e-02, 3.327421136445877e-02, -5.476680525804904e-16, 0.000000000000000e+00, -2.002318492394405e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
