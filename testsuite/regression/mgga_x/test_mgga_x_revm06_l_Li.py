
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revm06_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.904759644311025e+00, -1.327338450327835e+00, -1.483973868277602e-01, -1.564914116147631e-01, -4.644398146560202e-02, -9.131646074487361e-03, -1.553448575799858e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revm06_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.106256583944639e+00, -2.104963256158189e+00, -1.996485727741707e+00, -1.997510192092765e+00, -3.288266243859306e-01, -3.350594369223450e-01, -1.555329875661654e-01, -1.155353009732825e-02, -9.040513372976383e-02, -3.692236649070711e-04, -1.224023005397955e-02, -1.205530061474283e-02, -2.466249903604171e-04, -9.719341184218795e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revm06_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.008084510915096e-04, 0.000000000000000e+00, -4.989826340580931e-04, -2.131642126181639e-03, 0.000000000000000e+00, -2.124874773144971e-03, -2.595873756284390e-01, 0.000000000000000e+00, -2.658113705251229e-01, -7.109351337669417e+00, 0.000000000000000e+00, 7.447355217127062e-02, -2.012965880517775e+02, 0.000000000000000e+00, 4.838171996599130e-01, 3.221745742585475e-05, 0.000000000000000e+00, 7.060474567509273e-02, 2.208406895951217e-11, 0.000000000000000e+00, -1.558592312535517e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revm06_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.148150784902894e-02, -6.193763758754001e-02, 3.328507515751148e-02, 3.317472732687539e-02, 2.402118053404238e-02, 2.532441909761595e-02, -1.513161621210366e+00, -1.089673834831122e-05, 2.384606005388591e-01, -2.238961054695040e-09, -5.435431511330781e-09, -1.175878292231580e-05, -3.044736485371212e-20, 1.415015504808029e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
