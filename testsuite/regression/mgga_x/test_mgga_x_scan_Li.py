
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_scan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.975060568981262e+00, -1.303868411382315e+00, -2.342832281820834e-01, -1.809721028456183e-01, -5.246927185872651e-02, -4.816589062984131e-03, -3.519701192409019e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_scan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.759671175055252e+00, -2.762229466869101e+00, -2.007765339756655e+00, -2.008875749884981e+00, -3.254056935755857e-01, -3.259033195562962e-01, -2.474090534886361e-01, 1.887701663493957e+00, -7.766451651481042e-02, 6.782895887021723e+00, 4.112756009143705e+01, 1.874986943535569e+00, 6.727161134958854e+04, -1.712610651816098e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.844853033833104e-04, 0.000000000000000e+00, -3.830069690157897e-04, -2.653910025162050e-03, 0.000000000000000e+00, -2.639625617940376e-03, -4.331674351649799e-01, 0.000000000000000e+00, -4.336622701964421e-01, -4.761286701059610e+00, 0.000000000000000e+00, -4.860816314658011e+04, -2.001679957958974e+02, 0.000000000000000e+00, -1.375610938253658e+10, -1.867860488038170e+04, 0.000000000000000e+00, -4.135922010229128e+04, -3.092984265500826e+09, 0.000000000000000e+00, -4.411917456100856e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.817264550930526e-02, 1.815244838911764e-02, 3.951314439368471e-02, 3.941176371500422e-02, 2.378177799997487e-03, 2.508153711235688e-03, 1.717037773224435e-01, 6.211671012042239e-01, 6.451770599445764e-02, 5.604736975658233e+00, 2.774037471376389e-01, 6.012941965513501e-01, 3.755379456541990e-01, 2.839915194853458e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
