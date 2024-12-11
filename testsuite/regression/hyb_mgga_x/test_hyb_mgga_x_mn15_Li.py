
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_mn15_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([4.756877456529259e-01, -3.083768008183639e-01, -1.214417108151338e-01, 1.152115394350684e-01, 3.027580929399462e-03, 1.431150292365179e-02, 3.511532844538299e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_mn15_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.663748691161103e+00, 2.670663581493427e+00, 6.791624603803814e-01, 6.835112138240702e-01, -2.640251738894407e-01, -2.640558307620485e-01, 3.237756174369060e-01, 1.710243469682537e-02, -5.833326010083834e-02, 7.284341260649005e-04, 1.792079985432428e-02, 1.757992350004363e-02, 4.878904606206281e-04, 4.118217397028135e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn15_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.065774605753351e-03, 0.000000000000000e+00, -2.062900546883884e-03, -3.305103289976954e-03, 0.000000000000000e+00, -3.307865719512128e-03, -9.091602666974236e-01, 0.000000000000000e+00, -9.136657951683195e-01, -3.639635273928621e+00, 0.000000000000000e+00, 9.346469625220219e-03, -7.431968166268289e+02, 0.000000000000000e+00, 1.463761476476305e+00, -1.492939688602648e-06, 0.000000000000000e+00, -2.812562698136375e-04, 6.751385526383044e-11, 0.000000000000000e+00, -1.614090061565911e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn15_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.485617129892103e-01, -2.489926615631363e-01, -1.467604631090586e-01, -1.471606549309747e-01, 5.750623921177201e-03, 5.994880656107111e-03, -6.303767371612231e+00, 2.632897231497640e-05, 4.832241763695231e-01, 5.712495257468740e-09, 1.300033591347480e-08, 2.834408246321334e-05, 7.774196261055195e-20, 1.699474514460982e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
