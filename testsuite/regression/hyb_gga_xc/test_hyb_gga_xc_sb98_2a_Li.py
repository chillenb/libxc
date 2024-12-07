
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_2a_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.457900234692897e+00, -1.056595614064074e+00, -3.621542324631570e-01, -1.515503587189189e-01, -7.861085765149434e-02, -1.965443072757077e-02, -4.472152190344455e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_2a_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.838363217347313e+00, -1.839780121358645e+00, -1.287025830284890e+00, -1.287925330838636e+00, -3.348891364547092e-01, -3.352538069057684e-01, -1.883425161645966e-01, 2.028566191378329e-01, -7.230998223343324e-02, 1.417093376833096e-01, -2.625503572378043e-02, -2.579124053461355e-02, -7.264445522998144e-04, -2.238111130579328e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_2a_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.272614831297184e-04, 0.000000000000000e+00, -1.268608543241653e-04, -5.016715694595980e-04, 0.000000000000000e+00, -5.000730394300667e-04, -6.829026832951635e-02, 0.000000000000000e+00, -6.808656327238503e-02, -4.057115964915364e+00, 0.000000000000000e+00, 3.419533485449026e+01, -6.538732150242029e+01, 0.000000000000000e+00, 4.099809794622798e+03, -2.220923427035464e-01, 0.000000000000000e+00, -1.539870461531365e-01, -2.246863582801581e+00, 0.000000000000000e+00, 6.630027236033586e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
