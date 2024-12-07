
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn15_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.414260818930498e-01, -1.471142815563226e-01, 7.005251117172916e-02, -2.118750486838162e-02, 1.228036871377705e-02, -1.796651930943270e-02, -4.458778983416237e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn15_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.146632078469537e-01, 2.148677349689432e-01, -3.263364935278563e-01, -3.261257104857457e-01, 4.676169029731407e-02, 4.679326791828758e-02, 2.348401476850534e-02, -1.224074536507504e-01, -1.229870616101452e-02, 3.069971617949296e+00, -2.257752456446574e-02, -2.283094179622112e-02, -5.245294618793906e-04, -7.696734983522366e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.461415181789415e-05, -1.692283036357883e-04, -8.461415181789415e-05, -2.186131623227922e-04, -4.372263246455844e-04, -2.186131623227922e-04, 3.161583093370020e-02, 6.323166186740042e-02, 3.161583093370020e-02, -6.385892787148883e+00, -1.277178557429777e+01, -6.385892787148883e+00, 9.554530350373531e+01, 1.910906070074706e+02, 9.554530350373531e+01, 3.524560428017689e-03, 7.049120856194453e-03, 3.524560428017689e-03, 3.373860385305643e-05, 6.747731460900911e-05, 3.373860385305643e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_l_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.948597589192537e-02, -3.948597589192536e-02, 3.539863706945676e-02, 3.539863706945676e-02, -1.698095016424283e-02, -1.698095016424286e-02, -9.675689346027141e-01, -9.675689346025034e-01, -1.261223756032464e-01, -1.261223755163096e-01, -7.613802609947620e-08, -7.613802609947620e-08, -2.011643110611899e-19, -2.011933913207477e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
