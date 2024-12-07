
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_revapbeint_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.635108299798342e+01, 8.124369984141268e+00, 6.876337984867711e-01, 1.319055232519803e-01, 2.742704935727755e-02, 1.532845870680016e-03, 5.452312967211747e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_revapbeint_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.590910755458453e+01, 2.595676238757270e+01, 1.225019124523373e+01, 1.227156790078087e+01, 7.749444034517446e-01, 7.750541187874831e-01, 2.137132688487574e-01, 2.323560136149844e-03, 3.158156018328215e-02, 2.343131840869511e-06, 2.568631692283159e-03, 2.531746114232622e-03, 1.045386036845296e-06, 5.283293028718193e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_revapbeint_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.260022664186007e-03, 0.000000000000000e+00, 2.253853442471386e-03, 6.740081624920599e-03, 0.000000000000000e+00, 6.723156748454783e-03, 1.800337657832553e-01, 0.000000000000000e+00, 1.796321196707903e-01, 3.017604727614302e+00, 0.000000000000000e+00, 3.498191301634295e-02, 3.092015983938909e+01, 0.000000000000000e+00, 7.103625688000105e-03, 3.737816642046110e-02, 0.000000000000000e+00, 3.465251258600610e-02, 3.454054791974345e-03, 0.000000000000000e+00, 3.514821786485389e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
