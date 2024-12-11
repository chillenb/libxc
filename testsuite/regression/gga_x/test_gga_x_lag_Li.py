
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lag_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.742512848356008e+00, -1.216382224921721e+00, -3.491023681708568e-01, -1.572108185252186e-01, -6.812061445249533e-02, -8.975973471959191e-02, -3.635332053033939e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lag_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.300112841807417e+00, -2.302217644595994e+00, -1.583072408152169e+00, -1.584454600693756e+00, -3.101445212158804e-01, -3.098938440853967e-01, -2.086532608863938e-01, -2.496261413261134e-02, -7.283757225921556e-02, -5.346294152238623e-03, -2.571671850387783e-02, -2.576458030922307e-02, -5.344160625993718e-03, -4.707272095508712e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lag_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.804212738149206e-05, 0.000000000000000e+00, -3.788465494677573e-05, -2.006753423852640e-04, 0.000000000000000e+00, -1.999283103540591e-04, -7.531843341323888e-02, 0.000000000000000e+00, -7.535984894308943e-02, -4.825027636146338e-01, 0.000000000000000e+00, -8.943825848307733e+02, -3.936613532578515e+01, 0.000000000000000e+00, -3.288817488237550e+07, -7.766247671838560e+02, 0.000000000000000e+00, -7.783659636509109e+02, -9.706363679392435e+07, 0.000000000000000e+00, -2.880475191763287e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
