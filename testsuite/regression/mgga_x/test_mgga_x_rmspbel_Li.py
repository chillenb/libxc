
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rmspbel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.921759772187082e+00, -1.351987934465703e+00, -3.922094092856708e-01, -1.726975782265701e-01, -7.781158654272442e-02, -2.053487569831259e-02, -3.838585506945625e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rmspbel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.492634655077575e+00, -2.494970393257436e+00, -1.709730593683649e+00, -1.711460924388483e+00, -3.057743157669082e-01, -3.952269923696707e-01, -2.265642174341631e-01, -2.608213483510491e-02, -8.327350758377942e-02, -8.296405943693126e-04, -2.741778485069958e-02, -2.722272306143004e-02, -5.541548354971549e-04, -3.939539036233512e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmspbel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.176249070336938e-04, 0.000000000000000e+00, -1.170625321142806e-04, -4.876537367605624e-04, 0.000000000000000e+00, -4.828199852394270e-04, -1.791594112623349e-01, 0.000000000000000e+00, -6.227206346234682e-02, -1.920094267168326e+00, 0.000000000000000e+00, -4.925669667940220e-01, -4.480488187468799e+01, 0.000000000000000e+00, -3.158639759035284e+00, -5.010438138481877e-01, 0.000000000000000e+00, -4.673562674432135e-01, -2.299384621547234e+00, 0.000000000000000e+00, -3.291332783983968e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmspbel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([7.731066840235072e-06, 7.493944303257972e-12, 5.618445528739420e-05, 1.121911332859002e-17, 2.780577958714077e-02, 8.628839435619824e-11, 5.091063415793005e-03, 5.036685166460381e-19, 6.728934169175944e-07, 1.496644558038120e-16, 1.940530349010502e-19, 1.981121746219198e-19, 8.219955965319905e-37, 3.241417656662742e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
