
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b1wc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.564515446502717e+00, -1.115056787323510e+00, -3.362269508901622e-01, -1.494923537727515e-01, -6.556929731013286e-02, -1.724988978347341e-02, -3.224411826015363e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b1wc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.009652528333663e+00, -2.011297060188975e+00, -1.398370982604455e+00, -1.399408803640919e+00, -3.119122987368007e-01, -3.119272293640752e-01, -1.970189385140733e-01, -1.195524320923521e-01, -7.070374736161358e-02, 3.421216851398906e-01, -2.303476873306274e-02, -2.287050731459049e-02, -4.654900619621815e-04, -3.309212790834975e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b1wc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.451829438989384e-04, 9.190971700708733e-05, -1.445597396700080e-04, -5.427361126879479e-04, 2.980993506782570e-04, -5.406408722227663e-04, -6.874406553384424e-02, 6.249948659585063e-03, -6.866314166805279e-02, 2.870360580539844e-01, 6.762268918356340e+00, 2.989189014224693e+00, -3.576767152285904e+01, 2.258698854598489e+01, 8.641052031118249e+00, -3.961091737967286e-01, 3.357174600576258e-04, -3.707518026446399e-01, -1.931233676359047e+00, 3.212885779437900e-06, -2.764495725774416e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
