
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hjs_pbe_sol_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.402560192086155e+00, -9.971239200154868e-01, -3.056224617182194e-01, -1.476431765052193e-01, -6.933913366963229e-02, -2.045051297531889e-02, -3.838595910663383e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hjs_pbe_sol_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.834951064466081e+00, -1.836414083360395e+00, -1.280248633589831e+00, -1.281186124663075e+00, -3.023504834972039e-01, -3.023770440766121e-01, -1.932652892146248e-01, -1.235645631921081e-01, -7.343904729388885e-02, 3.419889428047509e-01, -2.722615686135282e-02, -2.703625496862308e-02, -5.541559468520087e-04, -3.939550388477289e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hjs_pbe_sol_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.591293716171252e-05, 9.190971700708733e-05, -6.552031413912854e-05, -3.101066534089802e-04, 2.980993506782570e-04, -3.085960894183348e-04, -5.116601297764992e-02, 6.249948659585063e-03, -5.108824660580255e-02, 1.495255557726503e+00, 6.762268918356340e+00, 2.895381546278569e+00, -3.538001445242506e+01, 2.258698854598489e+01, 8.142754513755360e+00, -4.923967331694586e-01, 3.357174600576258e-04, -4.600155986313650e-01, -2.291266879508307e+00, 3.212885779437900e-06, -3.276935422247640e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
