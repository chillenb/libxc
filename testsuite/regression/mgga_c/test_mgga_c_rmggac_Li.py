
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
    ref_tgt = [-2.565339837219293e-02, -2.438245090193295e-02, -1.449665727637643e-02, -2.724260402241904e-04, -1.371567822417214e-08, -7.957649910248841e-04, -3.756670063267504e-06]
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
    ref_tgt = [-2.821781469069264e-02, -2.814502164917611e-02, -2.858562995034522e-02, -2.852027210714631e-02, -2.500121630232035e-02, -2.502063770877238e-02, -5.928512975536369e-03, -1.628487211057751e-01, -9.087091675600290e-09, -8.050532872476401e-02, -1.508845280133769e-03, -1.522018117003338e-03, -6.828799792034531e-06, -9.204898282461700e-06]
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
    ref_tgt = [1.790176763892149e-06, 3.580353527784298e-06, 1.790176763892149e-06, 9.572333137373725e-06, 1.914466627474745e-05, 9.572333137373725e-06, 2.750705800493730e-03, 5.501411600987460e-03, 2.750705800493730e-03, 8.006246877374045e+00, 1.601249375474809e+01, 8.006246877374045e+00, 1.494206453628675e-05, 2.988412907257351e-05, 1.494206453628675e-05, 2.218472388834460e+00, 4.436944777668919e+00, 2.218472388834460e+00, 4.200359363129391e+03, 8.400718726258781e+03, 4.200359363129391e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rmggac_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
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
    ref_tgt = [-5.889588108189012e-08, -5.889588108189009e-08, -8.879464913757352e-07, -8.879464913757350e-07, -3.039773904020509e-04, -3.039773904020511e-04, -2.962802419757967e-01, -2.962802419757312e-01, -4.034968425852436e-10, -4.034968422592558e-10, 5.534563166749143e-09, 5.534563166749018e-09, 9.159755731501780e-18, 9.159755738041854e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
