
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rmggac_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.700145806999472e-02, -8.305310452030418e-02, -4.923612906978183e-02, -1.641673835967545e-02, -1.083086779353643e-02, -2.952311010867517e-04, -1.338090129662321e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rmggac_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.057935164342473e-02, -7.042583476933076e-02, -8.721570166848137e-02, -8.707778656206150e-02, -5.640287316170636e-02, -5.644621753925971e-02, -1.258677460448624e-02, -1.213133682583439e-01, -1.304681617141727e-02, -7.205716705986100e-02, -5.790710623900466e-04, -5.858633413738188e-04, -1.244177552421378e-07, -2.624203488445177e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rmggac_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.796621077421929e-04, 3.593242154843858e-04, 1.796621077421929e-04, 6.326656347713163e-04, 1.265331269542633e-03, 6.326656347713163e-04, 1.761908193140355e-01, 3.523816386280709e-01, 1.761908193140355e-01, 1.115941136766236e+01, 2.231882273532472e+01, 1.115941136766236e+01, 1.616698549557133e+02, 3.233397099114266e+02, 1.616698549557133e+02, 3.948417877596048e-02, 7.896835755192096e-02, 3.948417877596048e-02, -2.259652799794801e-03, -4.519305599589602e-03, -2.259652799794801e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rmggac_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.529225150358227e-03, -3.529225150358229e-03, -6.689448568879114e-04, -6.689448568879179e-04, 3.550258521352980e-05, 3.550258521351901e-05, -1.760567930279847e-01, -1.760567930279457e-01, 7.990396595008091e-04, 7.990396588551241e-04, -2.025223070324554e-07, -2.025223070324552e-07, 8.025541083729354e-13, 8.025541083729353e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
