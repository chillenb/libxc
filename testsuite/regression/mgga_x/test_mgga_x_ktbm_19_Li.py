
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_19_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.172951652945789e+00, -1.540060742715041e+00, -2.643273390926693e-01, -1.930704421274442e-01, -6.483103047191895e-02, -9.132936477933840e-03, -1.717030000035328e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_19_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.584185777943474e+00, -2.586668592477133e+00, -1.752854697720437e+00, -1.753985853738257e+00, -3.657795756454817e-01, -3.685708339898773e-01, -2.396312436101248e-01, -1.119758296640967e-02, -9.244346633416940e-02, -3.550286254625700e-04, -1.235148040281441e-02, -1.168959718722206e-02, -2.489322478881961e-04, -1.685838719896110e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_19_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.975432603165940e-04, 0.000000000000000e+00, -8.945004402130756e-04, -3.514363834963354e-03, 0.000000000000000e+00, -3.505228346696684e-03, -5.545130931658307e-02, 0.000000000000000e+00, -5.887888610016952e-02, -1.368313997425235e+01, 0.000000000000000e+00, -8.053740161462819e+00, -9.427045105843744e+01, 0.000000000000000e+00, -2.008284085272072e+04, 3.347719348068753e-01, 0.000000000000000e+00, -7.202951398336303e+00, 7.071223800471647e-01, 0.000000000000000e+00, -9.091732673669406e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_19_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_19_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_19", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.013941541234219e-02, 3.009575739218441e-02, 5.373409349611416e-02, 5.368332200473376e-02, 2.343318961860400e-02, 2.541068310507654e-02, 2.914105686645712e-01, 1.034711624637820e-04, 4.109679919707334e-01, 8.182584737134681e-06, -1.017430303262678e-07, 1.053151248398809e-04, -6.798391032221516e-16, 3.966117394729886e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
