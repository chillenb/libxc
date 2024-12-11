
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_wpbeh_whs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbeh_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.240849764657464e+00, -8.420001729745114e-01, -1.621940468557789e-01, -4.087773008031342e-02, -3.606959525409563e-03, -1.361066924314808e-05, -9.282305076866261e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_wpbeh_whs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbeh_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.636794355981797e+00, -1.638261885906218e+00, -1.082626956081631e+00, -1.083550525115543e+00, -1.664922031489085e-01, -1.666050672849801e-01, -6.990185082642139e-02, -9.766423145130051e-02, -1.044250090141339e-02, 3.428185824905136e-01, -2.764826003265109e-05, -2.704422736375305e-05, -2.235001531494462e-10, -8.030241258269285e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_wpbeh_whs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbeh_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.352631508713534e-04, 9.190971700708733e-05, -1.346423272069470e-04, -5.620879075356525e-04, 2.980993506782570e-04, -5.598397418629368e-04, -4.680024431006596e-02, 6.249948659585063e-03, -4.668337100096653e-02, 2.983062322509312e+00, 6.762268918356340e+00, 3.381127440951617e+00, 9.725551350246331e+00, 2.258698854598489e+01, 1.129349427298479e+01, 1.584426218153484e-04, 3.357174600576258e-04, 1.594120955445192e-04, 1.606542482281367e-06, 3.212885779437900e-06, 1.606543185015555e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
