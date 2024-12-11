
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lrc_wpbeh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.405983606863263e+00, -9.808870431284340e-01, -2.480301732516817e-01, -7.421892069734201e-02, -1.111165128757865e-02, -6.165891713190006e-05, -3.960537981990351e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lrc_wpbeh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.821630791915451e+00, -1.823211074905765e+00, -1.228624177023437e+00, -1.229623221256453e+00, -2.489221953977002e-01, -2.490627396656470e-01, -1.159484817418600e-01, -9.775002143192532e-02, -1.817596967349560e-02, 3.428185800405031e-01, -1.292948708999826e-04, -1.262067867256785e-04, -9.536029853865124e-10, -3.426219655085506e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lrc_wpbeh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.556319845000126e-04, 9.190971700708733e-05, -1.549371755132656e-04, -6.492145274360753e-04, 2.980993506782570e-04, -6.466602562237039e-04, -5.572988873787517e-02, 6.249948659585063e-03, -5.558715312229878e-02, 2.008546255264948e+00, 6.762268918356340e+00, 3.380798213881071e+00, -4.848777118095686e+00, 2.258698854598489e+01, 1.129349427285979e+01, -2.777917272808483e-04, 3.357174600576258e-04, -2.326932935536423e-04, 1.606524595270521e-06, 3.212885779437900e-06, 1.606536677235218e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
