
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r2scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.872327685557759e+00, -1.364193392521450e+00, -3.286291578446273e-01, -1.648943118745998e-01, -7.184056519569061e-02, -9.507971292321905e-03, -1.821094706455403e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_r2scanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.202166382361781e+00, -2.204302887662727e+00, -1.551044831301192e+00, -1.552193259213118e+00, -4.635909013777152e-01, -4.634039553401223e-01, -2.034452602963380e-01, -8.889800301207245e-03, -9.908290499170903e-02, "nan", -9.494703658949914e-03, "nan", -5.936388997857224e-05, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.047414665151078e-04, 0.000000000000000e+00, -5.023994914009044e-04, -1.407608323247062e-03, 0.000000000000000e+00, -1.404550700073566e-03, 1.227274353516568e-02, 0.000000000000000e+00, 1.227505756795533e-02, -8.021090882447291e+00, 0.000000000000000e+00, 1.599655796575493e+01, 7.211252344146685e+00, 0.000000000000000e+00, "nan", 1.499953757677561e+01, 0.000000000000000e+00, "nan", 4.131015009520217e+04, 0.000000000000000e+00, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [4.126623014666667e-03, 4.122195388993662e-03, 4.482827660608105e-03, 4.483727764884334e-03, 0.000000000000000e+00, 0.000000000000000e+00, 4.742820600215632e-02, 2.532462324931478e-04, 0.000000000000000e+00, "nan", 2.655876122105918e-04, "nan", 4.679999070318671e-09, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scanl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
