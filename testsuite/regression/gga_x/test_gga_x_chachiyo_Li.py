
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_chachiyo_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.801131636538278e+00, -1.289002219975512e+00, -4.442524433034828e-01, -1.605568302641694e-01, -8.192672013054612e-02, -1.847095247967052e-01, -6.972504516236014e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_chachiyo_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.245231699005596e+00, -2.247353455266138e+00, -1.525521326476958e+00, -1.526888266895032e+00, -3.066314338926744e-01, -3.064067573931998e-01, -2.052686000128039e-01, -4.625063309394338e-02, -7.007432101503663e-02, -1.267722128398716e-02, -4.675959206340573e-02, -4.724819198589253e-02, -1.215751688415367e-02, -1.045706820778815e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_chachiyo_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.660624722308895e-04, 0.000000000000000e+00, -2.651788149563443e-04, -1.014008501822692e-03, 0.000000000000000e+00, -1.010796615162296e-03, -1.385134796442541e-01, 0.000000000000000e+00, -1.385294595404610e-01, -4.280691140810521e+00, 0.000000000000000e+00, -1.891888273169014e+03, -8.569406033903311e+01, 0.000000000000000e+00, -6.207871307119742e+07, -1.648859417155034e+03, 0.000000000000000e+00, -1.649808831234243e+03, -1.822476741881132e+08, 0.000000000000000e+00, -5.399363041285005e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
