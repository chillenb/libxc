
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_b88b95_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b88b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.310055041399265e+00, -9.326858628204535e-01, -4.111334070892450e-01, -6.231008373614324e+79, -8.331796827303915e+80, -2.676263985520844e+109, -1.668954898256139e+56]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_b88b95_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b88b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.711573266611274e+00, -1.712938252248238e+00, -1.191855942081141e+00, -1.192717002775839e+00, -2.134230426524688e-01, -2.156261210638428e-01, 3.466989735193070e+66, -1.222528041916907e+81, 1.338913802901526e+71, -2.990940229289822e+85, -1.333313055500040e+107, -1.362532733770970e+107, -2.073705507148628e+54, -5.770395289772950e+40]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b88b95_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b88b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.405128169823690e-04, 0.000000000000000e+00, -1.402030933141903e-04, -2.385400772741077e-04, 0.000000000000000e+00, -2.395521888627716e-04, 4.636425573856521e+00, 0.000000000000000e+00, 4.532823171142924e+00, 1.524547848341678e+00, 0.000000000000000e+00, 5.649063800257172e+00, 2.265197018620901e+03, 0.000000000000000e+00, 6.632508543915026e+02, 1.193814816847380e-04, 0.000000000000000e+00, 8.771345115992075e-03, 5.497290808982469e-11, 0.000000000000000e+00, 3.225550174561746e+20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b88b95_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b88b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.873107274983154e-03, -2.871683944547766e-03, -5.332155022208856e-03, -5.324538201822170e-03, -5.428445283005290e-02, -5.432066785617735e-02, -1.392078133987678e-01, -1.176431803685671e-07, -5.378184861852190e-01, -3.855053355621430e-11, -5.743566782136254e-11, -1.256638198788993e-07, -5.361832595959912e-22, -6.047408678635331e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
