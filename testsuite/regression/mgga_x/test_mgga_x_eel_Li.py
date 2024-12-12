
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_eel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_eel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.038226830597969e+00, -1.414035485588661e+00, -3.197072241217540e-01, -1.841997647524651e-01, -7.173040112174026e-02, -6.441922808570396e-03, -2.628208077158291e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_eel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_eel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.709665877983303e+00, -2.712227201955952e+00, -1.733274280883170e+00, -1.734682808123835e+00, 8.239995482143299e-02, 7.993763541762897e-02, -2.456787038182913e-01, 4.473958317173745e-02, -2.647426788504297e-02, 5.478119161075362e-04, -1.157357873770390e-02, 4.690336253942507e-02, -5.715875079669833e-05, 1.184890920037693e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_eel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_eel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.237932821801900e-05, 0.000000000000000e+00, -3.126102933815005e-05, -2.206391509269604e-03, 0.000000000000000e+00, -2.196252266301348e-03, -6.861740422542513e-01, 0.000000000000000e+00, -6.895303056960820e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.399497661154355e+03, -4.173644518370774e+02, 0.000000000000000e+00, -1.301733523625768e+06, 2.450832777779904e+01, 0.000000000000000e+00, -1.257668253012355e+03, 3.991056242664525e+04, 0.000000000000000e+00, -2.905534128411879e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_eel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_eel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.728190457071839e-03, 1.658662267061408e-03, 3.912719074442636e-02, 3.899418013304599e-02, 1.664216464004354e-01, 1.690835046522711e-01, 0.000000000000000e+00, 1.822751267225574e-02, 1.017859948610136e+00, 5.408663396765168e-04, 0.000000000000000e+00, 1.863471697208998e-02, 0.000000000000000e+00, 1.304324557968707e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
