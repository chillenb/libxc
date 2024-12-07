
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_jk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.853114987251649e+00, -1.317575200176507e+00, -3.663669509208111e-01, -1.648717443664969e-01, "nan", -3.205799135244117e-02, -1.369539076114069e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_jk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.316737974516148e+00, -2.318038795618770e+00, -1.623994739148941e+00, -1.625347737377183e+00, -4.090133074928356e-01, -4.088732684780091e-01, -2.082218241813392e-01, -2.249233950823251e-02, "nan", "nan", -2.333334462051987e-02, -1.513748933946413e-02, -2.699610721673494e-03, -2.183783727950953e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.610098101632352e-04, 0.000000000000000e+00, -2.620712834924258e-04, -6.957488769490331e-04, 0.000000000000000e+00, -6.939454714012515e-04, -3.856430894588352e-02, 0.000000000000000e+00, -3.853960304024508e-02, -5.672288707602062e+00, 0.000000000000000e+00, -4.446715021084583e+02, -2.834823278282779e+01, 0.000000000000000e+00, "nan", -3.865603813194807e+02, 0.000000000000000e+00, 1.021121045174196e-163, -4.799731693621518e+07, 0.000000000000000e+00, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-2.332540603237351e-03, -2.304258287839297e-03, -2.281039295500965e-03, -2.280592831216083e-03, -1.996909503941568e-03, -1.993075450231363e-03, -3.303885676864351e-02, -1.109921289059678e-03, -1.450169764324545e-02, "nan", -1.123463193744021e-03, 0.000000000000000e+00, -1.076689980678513e-03, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
