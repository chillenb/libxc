
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revscan_vv10_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.707755030080274e-02, -2.553135770780387e-02, -1.309719987772237e-02, -3.061718063394891e-04, -4.784515158194613e-08, -1.111893021074258e-03, -6.455763794533750e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revscan_vv10_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.984523706141587e-02, -3.976839887910251e-02, -4.257655988348658e-02, -4.250811180660933e-02, -4.520296303730397e-02, -4.522043276094248e-02, -2.740407754888990e-03, -1.582640860487750e-01, -1.976712780717714e-02, -8.680614443578295e-02, -2.074541580592077e-03, -2.088576935462143e-03, -1.202888126651910e-05, -1.489224967766578e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revscan_vv10_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.434399257797813e-05, 4.868798515595624e-05, 2.434399257797813e-05, 9.362979741582605e-05, 1.872595948316521e-04, 9.362979741582605e-05, 1.658625800603576e-02, 3.317251601207152e-02, 1.658625800603576e-02, 3.610132138328515e+00, 7.220264276657028e+00, 3.610132138328515e+00, 1.153467080056535e+02, 2.306934160113071e+02, 1.153467080056535e+02, 3.111889937633177e+00, 6.223779875266354e+00, 3.111889937633177e+00, 7.218307235702503e+03, 1.443661447140501e+04, 7.218307235702503e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revscan_vv10_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revscan_vv10_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.307503569113028e-03, -2.307503569113027e-03, -2.809784119280092e-03, -2.809784119280092e-03, -6.832048288837600e-03, -6.832048288837597e-03, -1.366335957196810e-01, -1.366335957196508e-01, -2.758487640013235e-01, -2.758487637784634e-01, -7.885507414394226e-11, -7.885507414394227e-11, -6.740990255132599e-23, -6.740990255132604e-23]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
