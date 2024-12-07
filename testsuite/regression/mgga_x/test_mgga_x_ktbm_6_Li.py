
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_6_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.995905370152716e+00, -1.404546687663673e+00, -3.450689479144167e-01, -1.790308158936079e-01, -7.499983070300542e-02, -1.295676656348443e-02, -2.375307006875506e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_6_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.496448438622794e+00, -2.498885339580136e+00, -1.664564368894463e+00, -1.666037469428360e+00, -4.114693123229836e-01, -4.120995332324945e-01, -2.303277094323744e-01, -1.592363593594351e-02, -8.542158724962291e-02, -5.050397507551456e-04, -1.680713510390758e-02, -1.662296820700913e-02, -3.385255840791510e-04, -2.398161866327676e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_6_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.276262543147307e-04, 0.000000000000000e+00, -5.258088647507719e-04, -2.115303289627374e-03, 0.000000000000000e+00, -2.109216444236088e-03, -7.031570284453707e-02, 0.000000000000000e+00, -7.328772585401268e-02, -8.027045451939557e+00, 0.000000000000000e+00, -2.754183785007728e+01, -8.968260034922540e+01, 0.000000000000000e+00, -6.904147476150669e+04, -2.124705499564679e-01, 0.000000000000000e+00, -2.462435555347087e+01, -4.190569524962907e-01, 0.000000000000000e+00, -3.125620827073813e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_6_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_6_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.052450382257969e-02, 2.050898948084698e-02, 2.839365162179708e-02, 2.838807930481145e-02, 1.768615593380810e-02, 1.906574638892247e-02, 2.288185349694454e-01, 3.519138575820240e-04, 2.210951614335382e-01, 2.812997326985880e-05, 6.426860875646056e-08, 3.579495122996805e-04, 4.028864936449772e-16, 1.363495767820554e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
