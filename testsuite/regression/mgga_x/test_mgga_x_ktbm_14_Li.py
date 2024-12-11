
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_14_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.979779419785145e+00, -1.307482814130707e+00, -2.314597289971150e-01, -1.821848832972882e-01, -5.145136802411472e-02, -9.154123138598570e-03, -1.711719714075043e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_14_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.795779179030178e+00, -2.798470181144325e+00, -1.959626501131470e+00, -1.961264015576777e+00, -3.134649710125872e-01, -3.139441586625601e-01, -2.514554905834883e-01, -1.123376791857523e-02, -7.613153352782476e-02, -3.561813289993844e-04, -1.180975813954498e-02, -1.172736102841423e-02, -2.379084934747339e-04, -1.767004129468924e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_14_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.777389490172987e-04, 0.000000000000000e+00, -7.751101491553345e-04, -2.852064894819098e-03, 0.000000000000000e+00, -2.844778664760617e-03, -2.964226829859338e-02, 0.000000000000000e+00, -3.144736541341217e-02, -1.226069984359718e+01, 0.000000000000000e+00, -1.073554668518660e+01, -6.449087475503627e+01, 0.000000000000000e+00, -2.682379969605197e+04, -1.991656416730619e-01, 0.000000000000000e+00, -9.600274992501229e+00, -4.061864564415747e-01, 0.000000000000000e+00, 1.716177771168400e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_14_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.245121762767259e-02, 2.242040443633486e-02, 3.176375331349660e-02, 3.174813059872752e-02, 8.785301146760059e-04, 9.962102902704188e-04, 2.408348452269022e-01, 1.376944107957849e-04, 6.304592568560645e-02, 1.092908462270819e-05, 2.958131604021148e-06, 1.401167874335459e-04, 4.931755686896820e-11, -3.880961214727259e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
