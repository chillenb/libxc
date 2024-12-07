
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_vcml_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.865133372481342e+00, -1.329809893869288e+00, -4.054573904645915e-01, -1.665017317073855e-01, -8.115830887683732e-02, -9.533008390724226e-03, -1.451703973308882e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_vcml_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.344682925557918e+00, -2.346832169771551e+00, -1.599862323188248e+00, -1.600811287522948e+00, -1.025227091273863e-01, -4.698943230390740e-01, -2.143018431960045e-01, -1.467591799512431e-02, -8.120810023231254e-02, 9.916580930928921e+00, -1.038387288889147e-02, -1.533420105228329e-02, -2.066497148647183e-04, 4.513322658446577e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vcml_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.416411660967016e-04, 0.000000000000000e+00, -2.410846404364671e-04, -9.047514550152747e-04, 0.000000000000000e+00, -9.084635155998084e-04, -5.390230215225071e-01, 0.000000000000000e+00, -3.576590770112591e-02, -3.641363487738327e+00, 0.000000000000000e+00, 1.380361555636208e+00, -5.908834726831571e+01, 0.000000000000000e+00, -2.011204812732911e+10, 7.012775274693758e-01, 0.000000000000000e+00, 1.308805890896511e+00, 3.151573649866333e+00, 0.000000000000000e+00, -8.726644700115591e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vcml_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vcml_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.293512431741306e-05, -1.253514383840194e-11, -1.138880500386458e-04, -2.273069135258136e-17, 1.228605050205888e-01, 2.275899580849679e-10, -8.134660675164397e-03, 1.529669417213426e-12, -1.898846746979189e-06, 8.194360668902494e+00, 8.029565526357888e-16, 4.925383620046366e-13, 1.426902962231702e-33, 3.806841371325491e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
