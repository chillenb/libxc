
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_lm_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.155366642898562e-02, -4.763461939687870e-02, -4.284026302515943e-02, -5.933941262558625e-02, -1.648953295133718e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_lm_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.617962360696508e-02, -2.405123429728358e-01, -4.979251718172940e-02, -2.278944066631467e-01, -3.260495569503558e-02, -1.766334486038851e-01, 6.273788398145785e-02, -4.727229533440793e-02, 2.192021276176221e+00, 2.069044519388520e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_lm_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([8.228664174683485e-03, 2.974605272497509e-02, -1.323236291316922e+13, -1.056397457578754e-03, 1.844605330988462e-02, -1.323276616020483e+13, -3.575454282671098e-02, 1.041554629948939e-01, -1.323262925226530e+13, -1.618631283080286e+01, 2.162869825911018e+00, -1.323266749474097e+13, -3.508807186826470e+06, -6.383295925876493e+03, -1.323266817002347e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
