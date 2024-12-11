
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pbe2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.077481897827612e+01, 1.221172963934992e+01, 2.153620907558637e+00, 1.534279115641361e-01, 7.425250028798651e-02, 5.395950886792269e-03, 1.919390709981889e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pbe2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.209824438584175e+01, 2.214459550875396e+01, 9.225234824416273e+00, 9.243582891913562e+00, 1.590676340171035e+00, 1.595065194969888e+00, 1.939830981794223e-01, 8.179087301305118e-03, 3.511193070542522e-02, 8.248581508179752e-06, 9.041677021058733e-03, 8.911874380088225e-03, 3.680097184483053e-06, 1.859890187868197e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pbe2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.142406141265788e-02, 0.000000000000000e+00, 2.136974667439274e-02, 5.855418572875070e-02, 0.000000000000000e+00, 5.841676956431589e-02, 9.690736151530819e-01, 0.000000000000000e+00, 9.661341085496853e-01, 3.007680991315000e+01, 0.000000000000000e+00, 1.267012772948883e-01, 1.939698973581446e+02, 0.000000000000000e+00, 2.569955202989535e-02, 1.353979230957501e-01, 0.000000000000000e+00, 1.255169973683014e-01, 1.249608381701424e-02, 0.000000000000000e+00, 1.271592121365714e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
