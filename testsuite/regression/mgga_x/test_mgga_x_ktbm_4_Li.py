
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.916601964805750e+00, -1.279551600023008e+00, -2.535390634909699e-01, -1.758221001907765e-01, -5.501810492189159e-02, -1.180057825264867e-02, -2.146417936101218e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.678080076030355e+00, -2.680566555108478e+00, -1.866841157455716e+00, -1.868438808154821e+00, -3.343666468865462e-01, -3.344700155982655e-01, -2.413790079261306e-01, -1.264730017436993e-02, -7.735690345679938e-02, -4.011147888681776e-04, -1.329952060156184e-02, -1.320276627393756e-02, -2.679216382067543e-04, -1.973541982287481e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.365091580588158e-04, 0.000000000000000e+00, -3.353244035273419e-04, -1.311305781139018e-03, 0.000000000000000e+00, -1.307548411025575e-03, -4.033096103167604e-02, 0.000000000000000e+00, -4.181689978706245e-02, -5.192966825689886e+00, 0.000000000000000e+00, -6.069942585947266e+01, -5.218010137637237e+01, 0.000000000000000e+00, -1.523723524391617e+05, -1.131251192123493e+00, 0.000000000000000e+00, -5.426492577450006e+01, -2.307365992335028e+00, 0.000000000000000e+00, -1.477737724165543e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_4_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.764446471818915e-02, 1.762754910182209e-02, 2.363332032828225e-02, 2.362627522048186e-02, -6.890028755281185e-04, -6.473456613297374e-04, 1.956335823946531e-01, 7.753372725792024e-04, 3.348700072840920e-02, 6.208191341247400e-05, 1.680045801106111e-05, 7.885519392559533e-04, 2.801512747071772e-10, -6.038685214405115e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
