
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_sg4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.516573561483509e-02, -3.692578034867207e-02, -4.163374860961591e-04, -1.389628899020666e-02, -1.848455317532416e-03, -7.347175175993498e-05, -1.681738173367795e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_sg4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.161845902567351e-01, -1.160811662688824e-01, -1.006039887826006e-01, -1.005489261187467e-01, -3.147536028373964e-03, -3.053598552874545e-03, -2.590132970822979e-02, -9.015942556784191e-02, 7.122651874225680e-04, 1.255898138583557e+00, -1.303915164856757e-02, 1.365720185467361e-02, -1.978391892536515e-04, -2.903012928114558e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_sg4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.020017004472030e-05, 1.004003400894406e-04, 5.020017004472030e-05, 1.623118863462617e-04, 3.246237726925234e-04, 1.623118863462617e-04, 6.557439033512247e-04, 1.311487806702449e-03, 6.557439033512247e-04, 5.018755782245128e+00, 1.003751156449026e+01, 5.018755782245128e+00, -7.657910680719901e+00, -1.531582136143980e+01, -7.657910680719901e+00, -1.209571846060121e+00, -2.419143692120241e+00, -1.209571846060121e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
