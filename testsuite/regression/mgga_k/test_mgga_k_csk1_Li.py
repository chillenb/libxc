
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_csk1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.641287739236541e+01, 8.550224002340810e+00, 3.248072993831936e+00, 1.328802206036000e-01, "nan", 1.176944135381742e+166, 8.879178266767201e+257]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_csk1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.531558682075869e+01, 2.533202624550868e+01, 9.330604301571199e+00, 9.354814806353055e+00, -3.075003481834112e+00, -3.085183759501459e+00, 2.136219668354994e-01, -3.057065316931280e+00, "nan", "nan", -6.930966674342957e+150, 5.674799568746964e+150, 6.452647315997695e+242, 2.309007955429813e+243]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.386049160059192e-03, 0.000000000000000e+00, 3.429323564879311e-03, 2.586446108302639e-02, 0.000000000000000e+00, 2.576619460460348e-02, 4.113627147872595e+00, 0.000000000000000e+00, 4.119168564115552e+00, 3.028277494931346e+00, 0.000000000000000e+00, 7.829811389136195e+04, 4.027411056959991e+02, 0.000000000000000e+00, "nan", 6.733474560259016e+04, 0.000000000000000e+00, 7.647285960080892e+03, 8.236143114997581e+09, 0.000000000000000e+00, 2.547064861199871e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk1_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.545565353848909e-01, 1.540452429330453e-01, 1.032815457360462e-01, 1.033871045633374e-01, 1.795796339780856e-03, 1.782779030504907e-03, 1.657036488450410e-01, 4.373576373113266e-09, 6.910173990240943e-03, "nan", 5.436096555973936e-09, 1.666666666666668e-01, 4.084293677517660e-15, 1.666666666666666e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk1_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
