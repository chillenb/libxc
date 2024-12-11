
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_jrgx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_jrgx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.047384464915130e-02, -5.494908482011330e-02, -7.071850425784378e-03, -1.636667082645073e-02, -2.813612665765483e-03, -2.403739875027344e-08, -5.718895409317915e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_jrgx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_jrgx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.172999365708075e-01, -1.171600599482059e-01, -1.065061023538062e-01, -1.063954532117641e-01, -3.170596996831259e-02, -3.171780820131911e-02, -2.308072404056464e-02, -1.095860818509676e-01, -1.136558323103356e-02, 5.207527717837055e-01, -1.555130866285924e-07, -1.562943757245116e-07, -3.595736176257441e-15, -4.257018186310663e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_jrgx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_jrgx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.687400323784305e-05, 7.374800647568610e-05, 3.687400323784305e-05, 1.288307730576733e-04, 2.576615461153466e-04, 1.288307730576733e-04, 5.967206215007410e-03, 1.193441243001482e-02, 5.967206215007410e-03, 2.086015079944714e+00, 4.172030159889427e+00, 2.086015079944714e+00, 1.821686126571499e+01, 3.643372253142999e+01, 1.821686126571499e+01, 5.299980805594957e-04, 1.059996161117550e-03, 5.299980805594957e-04, 5.079214754686650e-06, 1.015838342300982e-05, 5.079214754686650e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
