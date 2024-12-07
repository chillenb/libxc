
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_wpbe08_whs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe08_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.608219861651897e+00, -1.082425118427894e+00, -1.953182491157725e-01, -4.398765335352424e-02, -3.707545121271928e-03, -1.430956683197512e-05, -9.778943064584996e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_wpbe08_whs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe08_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.118445092325595e+00, -2.120440619370029e+00, -1.385859933419040e+00, -1.387119152180498e+00, -1.981259905996673e-01, -1.982648459231216e-01, -7.648273781456226e-02, -9.766543161015308e-02, -1.077864856679737e-02, 3.428185824503964e-01, -2.903051160137719e-05, -2.839855322631591e-05, -2.354492428545516e-10, -8.461654920511137e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_wpbe08_whs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe08_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.924867072175846e-04, 9.190971700708733e-05, -1.916721087143163e-04, -7.825309074584023e-04, 2.980993506782570e-04, -7.796000636456713e-04, -6.067622725168770e-02, 6.249948659585063e-03, -6.052896053167524e-02, 2.987263798395041e+00, 6.762268918356340e+00, 3.381129572660795e+00, 1.003812478902051e+01, 2.258698854598489e+01, 1.129349427298608e+01, 1.613124273456966e-04, 3.357174600576258e-04, 1.619853287459266e-04, 1.606542669932452e-06, 3.212885779437900e-06, 1.606543253590986e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
