
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_exp4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_exp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.556133755798107e+01, 9.916554736664338e+00, 9.163338954773601e-01, 1.163436632346801e-01, 4.235960743923446e-02, 1.420881596769134e-03, 5.048676299579706e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_exp4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_exp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.563588588788852e+01, 1.568570921489033e+01, 2.172037195222605e+00, 2.183742823377539e+00, 1.527872367412802e+00, 1.526573121870285e+00, 1.633188684922646e-01, 2.157028307526469e-03, 7.059934763553577e-02, 2.169681079670840e-06, 2.385231635792612e-03, 2.350661847113112e-03, 9.679972614794755e-07, 4.892171903515000e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_exp4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_exp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.761950276125012e-02, 0.000000000000000e+00, 1.754702786538313e-02, 7.558324901639490e-02, 0.000000000000000e+00, 7.538170625334401e-02, 1.063982040828247e-34, 0.000000000000000e+00, 5.648030011422595e-35, 1.491423574749755e+01, 0.000000000000000e+00, 0.000000000000000e+00, 2.215400074101847e-05, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
