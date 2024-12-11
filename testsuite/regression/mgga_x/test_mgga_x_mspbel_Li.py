
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mspbel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.861704174015606e+00, -1.239289626868440e+00, -2.643111879621778e-01, -1.701354939514651e-01, -5.788587100205460e-02, -2.054607074459901e-02, -3.450789298810724e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mspbel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.588309595336653e+00, -2.590700421544033e+00, -1.817601973024349e+00, -1.819506895476759e+00, -3.572992445515785e-01, -3.574838551293583e-01, -2.307968438172038e-01, -2.608213483510486e-02, -8.150191024866647e-02, -8.296405943696783e-04, -2.750624612241474e-02, -2.722272306143002e-02, -5.541564189161880e-04, -1.983334294199957e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.244129917919258e-04, 0.000000000000000e+00, -4.227023526072927e-04, -2.026461764343760e-03, 0.000000000000000e+00, -2.022952480260183e-03, -2.466930990396606e-01, 0.000000000000000e+00, -2.472701923096796e-01, -4.751263367332583e+00, 0.000000000000000e+00, -4.925669667953466e-01, -1.251516796381998e+02, 0.000000000000000e+00, -3.158639228738505e+00, -2.103894862912166e-04, 0.000000000000000e+00, -4.673562674436461e-01, -2.727866099325800e-08, 0.000000000000000e+00, -2.393660790669036e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.528861568589373e-02, 1.526547811995671e-02, 2.428024148858703e-02, 2.432400531655886e-02, 8.825105540825775e-04, 9.492959081834605e-04, 1.104616813373910e-01, 1.759412844824860e-17, 3.617738711671697e-02, 2.557717309786170e-18, -2.880952997694070e-20, 6.268541651888014e-18, 3.294562551433540e-18, 1.080666026151967e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
