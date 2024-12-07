
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pw91_mod_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.802292281768179e+00, -1.290407603983884e+00, -4.191030146225423e-01, -1.605409548305722e-01, -8.051009952727987e-02, -3.174853753615858e-04, -4.457470496142169e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pw91_mod_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.242987917400967e+00, -2.245106263667802e+00, -1.526437438221461e+00, -1.527795583368132e+00, -3.842272501063934e-01, -3.843723095543455e-01, -2.050857053517769e-01, -1.134464613830592e-03, -7.602534660686548e-02, -8.292522446018547e-08, -1.335782779508540e-03, -1.260313768124915e-03, -2.102238742399720e-08, -9.026624051407569e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pw91_mod_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.725610339293989e-04, 0.000000000000000e+00, -2.716622847586327e-04, -1.019051434081494e-03, 0.000000000000000e+00, -1.015875636988883e-03, -8.465739116150517e-02, 0.000000000000000e+00, -8.448971864014498e-02, -4.398697499338729e+00, 0.000000000000000e+00, 7.341864520888815e+00, -6.853867099823904e+01, 0.000000000000000e+00, 4.208407419027946e+01, 7.497897831591406e+00, 0.000000000000000e+00, 6.986044605682785e+00, 3.053549549117265e+01, 0.000000000000000e+00, 4.365482522726935e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
