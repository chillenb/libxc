
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_l04_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.634802266721564e+01, 8.131254279823533e+00, 7.633061563178585e-01, 1.318884079582167e-01, "nan", 1.233052847301784e-03, 4.381283454127938e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_l04_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.590120791741867e+01, 2.594785335393287e+01, 1.216747025616654e+01, 1.218902754902220e+01, 1.102510418586640e+00, 1.103226413542984e+00, 2.137390357962170e-01, 1.871886492239002e-03, "nan", "nan", 2.069922989222296e-03, "nan", 8.400361072293508e-07, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l04_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.265402882889849e-03, 0.000000000000000e+00, 2.260930972290599e-03, 7.236460679850286e-03, 0.000000000000000e+00, 7.216800275620034e-03, 8.246306622074445e-02, 0.000000000000000e+00, 8.164813726366350e-02, 2.991256106964204e+00, 0.000000000000000e+00, 5.022318067777106e-06, 4.974771279551312e+01, 0.000000000000000e+00, "nan", 5.985253659350439e-06, 0.000000000000000e+00, "nan", 4.779445535921869e-10, 0.000000000000000e+00, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l04_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.312151465412201e-03, -1.331240507838879e-03, -2.716529711545534e-03, -2.713259339484135e-03, -1.103763433072230e-03, -1.091444809406163e-03, -7.747258447576962e-04, -3.766748379099338e-12, -6.585703472358718e-03, "nan", -5.219701279914729e-12, 0.000000000000000e+00, -3.408524774591217e-21, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l04_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
