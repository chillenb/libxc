
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_dk87_r2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.797690389641529e+00, -1.291699056648753e+00, -4.176203261556513e-01, -1.600057299012815e-01, -8.047346312086158e-02, -1.310738211003461e-01, -7.384126521068035e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_dk87_r2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.221061208182741e+00, -2.223195348843136e+00, -1.506476501553123e+00, -1.507812336780726e+00, -3.729993902125720e-01, -3.728923575821566e-01, -2.040719960125072e-01, -1.811299400912189e-02, -7.815119355410145e-02, -2.351191542767450e-03, -1.887177617171919e-02, -1.881164529683122e-02, -2.328803391975076e-03, -2.055295728539663e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_dk87_r2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.996373437476938e-04, 0.000000000000000e+00, -2.986051036951954e-04, -1.133313483068837e-03, 0.000000000000000e+00, -1.129844224595099e-03, -8.914012706708506e-02, 0.000000000000000e+00, -8.909599645325671e-02, -4.516611878513992e+00, 0.000000000000000e+00, -1.489684635012336e+03, -6.377989578883873e+01, 0.000000000000000e+00, -6.990250407421561e+07, -1.287771067146718e+03, 0.000000000000000e+00, -1.293259131315775e+03, -2.145890421971947e+08, 0.000000000000000e+00, -6.500116693718044e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
