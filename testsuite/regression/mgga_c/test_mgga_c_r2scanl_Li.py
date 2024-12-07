
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.965801390648251e-02, -3.185372393017430e-02, -1.449166625055379e-02, -1.226253420987959e-02, 2.802618331854134e-09, -5.846194648235521e-04, -5.302292720117550e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.140321070703213e-01, -1.139476890730553e-01, -8.808513776954874e-02, -8.807040110771581e-02, -2.402943348233415e-02, -2.398647293548019e-02, -2.546473431230963e-02, 2.849744869225731e-01, -2.059369741071328e-08, "nan", 3.060298623875582e-03, "nan", -3.343553783829048e-07, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.631340558964783e-05, 1.017318154756642e-04, 5.626992557957601e-05, 1.462550644086471e-04, 2.886809812814018e-04, 1.463319578018899e-04, -1.139563570643605e-02, 3.126000505282069e-02, -1.143014477398195e-02, 5.670861572920242e+00, 1.056677417867587e+01, -1.051364669229721e+04, -2.362398192128962e-05, 1.567388404447825e+02, "nan", -9.212100038234253e+01, 1.989691900543284e+02, "nan", 4.347534848282320e+02, 1.861132498778139e+03, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-5.979416608566208e-04, -5.976498608640978e-04, -7.196877827858584e-04, -7.199557953651613e-04, 0.000000000000000e+00, 0.000000000000000e+00, -2.796037319957428e-02, -4.763417908923591e-04, 0.000000000000000e+00, "nan", -1.120297845925238e-05, "nan", -2.136366495816226e-13, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scanl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
