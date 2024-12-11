
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_thakkar_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.627735637371194e+01, 8.146142750437685e+00, 6.954713137429525e-01, 1.307001209466676e-01, 2.751317624470452e-02, 9.536297476547030e-03, 7.332211852544510e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_thakkar_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.536738900349835e+01, 2.541389848693067e+01, 1.207054208600296e+01, 1.209137607715863e+01, 7.218553642844517e-01, 7.214859622299804e-01, 2.091891068343963e-01, 5.289442039169829e-03, 3.194165573936990e-02, 5.616556083504225e-05, 5.633539253775980e-03, 5.646221504601824e-03, 3.795320956839512e-05, 2.386132645501069e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_thakkar_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.978348762938581e-03, 0.000000000000000e+00, 2.971019453074608e-03, 7.877962951776349e-03, 0.000000000000000e+00, 7.859879582926185e-03, 2.120856290518865e-01, 0.000000000000000e+00, 2.119329970507579e-01, 4.239061382756558e+00, 0.000000000000000e+00, 9.307482737856628e+01, 3.044629637536069e+01, 0.000000000000000e+00, 1.068825905941996e+05, 8.510861767719196e+01, 0.000000000000000e+00, 8.462449113169806e+01, 2.119534254859573e+05, 0.000000000000000e+00, 4.488580561403669e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
