
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_csk_loc1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([4.561811368160446e+00, 6.403280198842597e+00, 3.456691901520546e+00, 5.889840576863213e-02, 7.742104502892838e-02, 6.790618985826407e+01, 1.144554864212277e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_csk_loc1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.075067645723642e+00, -3.070903404959763e+00, 3.563606912771605e+00, 3.578329580098562e+00, -1.325775626010000e-01, -8.914579147824007e-01, 9.520862256028778e-02, 5.054534597553485e-01, -4.127792743601590e-02, -1.210227402334236e+00, 5.014518844872965e-01, 5.188933394301501e-01, 2.340442352434578e-01, -1.185586587634537e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.925080852778589e-02, 0.000000000000000e+00, 1.919879096397346e-02, 4.559641528237266e-02, 0.000000000000000e+00, 4.546507455711455e-02, 1.017950367940041e+00, 0.000000000000000e+00, 1.902362005866516e+00, 2.105716507598984e+01, 0.000000000000000e+00, -1.291918909342438e+04, 3.412264720705336e+02, 0.000000000000000e+00, 2.454376727173730e+09, -1.111023334654098e+04, 0.000000000000000e+00, -1.135621965072012e+04, -1.358963613974630e+09, 0.000000000000000e+00, 2.292358375079806e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc1_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([2.042813408713162e-04, 2.050057298759011e-04, 3.879690391742802e-02, 3.884593838822663e-02, 1.406954992289786e-01, 1.011186750488823e-01, 3.572286092495014e-02, 2.171249999999520e-01, 3.428618804727154e-02, 2.092514915174386e-13, 2.171250000000001e-01, 2.171250000000002e-01, 2.171249999999998e-01, 6.278697865278834e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
