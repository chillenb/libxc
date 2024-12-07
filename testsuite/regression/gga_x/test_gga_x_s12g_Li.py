
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_s12g_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.811056432974672e+00, -1.280861061896330e+00, -4.262256770678194e-01, -1.631265034676636e-01, -8.291746842193241e-02, -2.001293290238877e-02, -3.738580011480452e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_s12g_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.359870921956048e+00, -2.362181270513362e+00, -1.540680920176977e+00, -1.542244440636535e+00, -4.295731232879801e-01, -4.298066799915974e-01, -2.162805002917156e-01, -2.544827942711659e-02, -7.771426404544268e-02, -8.080294911521859e-04, -2.675706035174680e-02, -2.656407954527854e-02, -5.397182615520144e-04, -3.836905424080107e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_s12g_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.220314381269692e-05, 0.000000000000000e+00, -9.164098290647656e-05, -8.769398616628467e-04, 0.000000000000000e+00, -8.731491112727331e-04, -6.728600250236585e-02, 0.000000000000000e+00, -6.706981466114717e-02, -6.254500607011005e-01, 0.000000000000000e+00, -1.876616297822261e-01, -7.186677216132486e+01, 0.000000000000000e+00, -1.199601252450225e+00, -1.907188661042421e-01, 0.000000000000000e+00, -1.780914378490579e-01, -8.732645629124653e-01, 0.000000000000000e+00, -1.249988055864346e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
