
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_gam_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.062983289275057e-03, -7.254760395990823e-03, -2.444548979517482e-02, 1.032705037590332e-02, -6.914779699624141e-03, 4.827805603917291e-02, 8.190700694174063e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_gam_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.597771596231138e-02, -2.604704797528562e-02, 5.119471343114156e-02, 5.126734515634251e-02, -2.155038692240426e-01, -2.166237441689093e-01, -1.562585883400257e-02, 1.856069366579409e+00, 4.282670457026862e-03, 1.145677153293155e+00, 6.026168104589417e-02, 6.213573496463243e-02, 5.029079208037350e-04, 2.728926762734464e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_gam_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.073138872798804e-05, 0.000000000000000e+00, 4.095912487162286e-05, -3.117708937207282e-04, 0.000000000000000e+00, -3.104455281418967e-04, 9.097930890243924e-02, 0.000000000000000e+00, 9.135315576446391e-02, 1.315481212913690e+01, 0.000000000000000e+00, 4.848623437545393e+02, -2.746549240231727e+01, 0.000000000000000e+00, 5.738833863631138e+04, 5.867696317155210e+00, 0.000000000000000e+00, 6.224954450412943e+00, 9.639178881858243e+00, 0.000000000000000e+00, 1.514775384479787e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
