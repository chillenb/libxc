
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.064081791118127e+00, -1.446746890458712e+00, -3.288650680684294e-01, -1.852266772173453e-01, -7.290523961307266e-02, -1.234520778868394e-02, -2.285675590720188e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.571816649399722e+00, -2.574322529044242e+00, -1.717552756815253e+00, -1.719014289568938e+00, -4.029927471011433e-01, -4.040018738901011e-01, -2.375753376564069e-01, -1.518163660916647e-02, -8.671256024828541e-02, -4.814961153982443e-04, -1.627950500111936e-02, -1.584840248537442e-02, -3.279737619295777e-04, -2.286365733281924e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.300745157467495e-04, 0.000000000000000e+00, -6.279289963062368e-04, -2.487471255322707e-03, 0.000000000000000e+00, -2.480602475066733e-03, -6.520624858252475e-02, 0.000000000000000e+00, -6.838912739743865e-02, -9.631013468938201e+00, 0.000000000000000e+00, -1.961391763937066e+01, -9.101748494169172e+01, 0.000000000000000e+00, -4.911805279280082e+04, 3.820179819106722e-02, 0.000000000000000e+00, -1.753732954169499e+01, 9.738902674780686e-02, 0.000000000000000e+00, -2.223650181639738e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.682470406459213e-02, 2.680220611920238e-02, 3.825199024634630e-02, 3.824203986840496e-02, 1.816190828661994e-02, 1.969313574443396e-02, 2.936769660911599e-01, 2.507221489583975e-04, 2.603961204002570e-01, 2.001247825755235e-05, -1.186858394995256e-08, 2.550448890625306e-04, -9.363289431551724e-17, 9.700275680683030e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
