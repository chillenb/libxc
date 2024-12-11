
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw1pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.412359719031530e+00, -1.012109218356749e+00, -3.233123609301170e-01, -1.356049121255765e-01, -6.211626060040322e-02, -1.284064278642284e-03, -4.920033032098508e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw1pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.797589494043716e+00, -1.799052857357693e+00, -1.245373226203777e+00, -1.246297790488770e+00, -2.852176470018171e-01, -2.852114662079414e-01, -1.779945612565020e-01, -1.018282466505392e-01, -6.254350654180357e-02, 3.428178651903981e-01, -4.844025256832787e-03, -4.607227120013642e-03, -2.083764338225116e-07, -9.599976767994827e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw1pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.610927817687092e-04, 9.190971700708733e-05, -1.604099017193237e-04, -6.259620712475124e-04, 2.980993506782570e-04, -6.235436507962617e-04, -7.333395807948467e-02, 6.249948659585063e-03, -7.326778471787371e-02, 3.964464667267942e-02, 6.762268918356340e+00, 2.903938979021169e+01, -4.354944584621194e+01, 2.258698854598489e+01, 3.570363151104774e+02, 2.572653996525570e+01, 3.357174600576258e-04, 2.417054089937557e+01, 2.871433999987504e+02, 3.212885779437900e-06, 4.404424311735985e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
