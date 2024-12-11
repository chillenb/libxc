
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbrxc_bg_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxc_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.713434035644613e+00, -1.195065672750865e+00, -3.352255000237369e-01, -1.563429226782719e-01, -6.597931183440675e-02, -8.931819558377880e+01, -3.497602310616741e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbrxc_bg_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxc_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.270835656228672e+00, -2.273010504887029e+00, -1.573954341899407e+00, -1.575309510863544e+00, -3.674262530825692e-01, -3.666690838079204e-01, -2.055011355865996e-01, 5.472551312118537e+00, -7.815232642732163e-02, 4.575205367706604e+01, 1.928207411079070e+02, 5.406591099852224e+00, 5.883234898739695e+06, -1.377335468019097e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxc_bg_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxc_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.708726353858998e-04, 0.000000000000000e+00, -2.694936876785443e-04, -1.245428695456661e-03, 0.000000000000000e+00, -1.240792581155933e-03, -4.773102805014515e-01, 0.000000000000000e+00, -4.840838894479554e-01, -3.937308489645173e+00, 0.000000000000000e+00, -1.204030721171263e+05, -2.702220476580300e+02, 0.000000000000000e+00, -7.496071763259512e+10, -7.179087900992873e+04, 0.000000000000000e+00, -1.019818187002092e+05, -2.107305668227600e+11, 0.000000000000000e+00, -1.481120558720395e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxc_bg_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxc_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.785632681696279e-03, -1.782409024675656e-03, -2.749362841110653e-03, -2.746128446933732e-03, -1.460854477910504e-02, -1.479696340154562e-02, -1.921307724017847e-02, -2.507395974295670e-05, -8.214808816167844e-02, -1.265501229657029e-06, -3.963853381170653e-07, -2.565065856846721e-05, -3.195516539836848e-12, -8.209435859072658e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
