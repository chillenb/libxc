
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zvpbesol_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.703036616386247e-02, -5.108202438665140e-02, -5.363725291275163e-03, -1.605569729352310e-02, -5.162089256399317e-03, -2.112840167579899e-08, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zvpbesol_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.173244393314780e-01, -1.171889568134567e-01, -1.054179945937674e-01, -1.053125124400383e-01, -2.548144777875537e-02, -2.549076712088638e-02, -2.322714385293650e-02, -1.059504266902033e-01, -2.879573931288687e-03, 2.470725754045352e-01, -2.176315152409718e-06, 2.036033609076952e-06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zvpbesol_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.028547001493514e-05, 8.057094002987029e-05, 4.028547001493514e-05, 1.373922762263564e-04, 2.747845524527128e-04, 1.373922762263564e-04, 4.882146493287788e-03, 9.764292986575581e-03, 4.882146493287788e-03, 2.334089543681316e+00, 4.668179087362633e+00, 2.334089543681316e+00, -1.027756761676635e+01, -2.055513523353270e+01, -1.027756761676635e+01, 2.680753773036527e-04, 5.361507546298707e-04, 2.680753773036527e-04, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
