
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lambda_lo_n_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.794390152295906e+00, -1.283600144998251e+00, -4.160595595717208e-01, -1.600245812720761e-01, -8.050393831203528e-02, -2.054541146396936e-02, -3.838759960700823e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lambda_lo_n_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.242610487192540e+00, -2.244744229068217e+00, -1.518988847707419e+00, -1.520357903652451e+00, -4.007575237757509e-01, -4.009342782617965e-01, -2.053102111008783e-01, -2.611690533740182e-02, -7.639962343206523e-02, -8.296807070906021e-04, -2.745847085578047e-02, -2.726116782091207e-02, -5.541805010100054e-04, -3.939719546945028e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lambda_lo_n_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.551531907499465e-04, 0.000000000000000e+00, -2.542743409581703e-04, -1.010492333840917e-03, 0.000000000000000e+00, -1.007250036141865e-03, -7.467995027371091e-02, 0.000000000000000e+00, -7.449564811972942e-02, -3.950162149345480e+00, 0.000000000000000e+00, -2.777726038845564e-01, -6.770168962110607e+01, 0.000000000000000e+00, -1.776811945521487e+00, -2.822761233896261e-01, 0.000000000000000e+00, -2.635961202890214e-01, -1.293453590123321e+00, 0.000000000000000e+00, -1.851445738546785e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
