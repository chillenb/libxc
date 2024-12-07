
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_csk_loc4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.520366626337600e+01, 6.972941013813906e+00, 3.215777615556111e+00, 1.273262447182655e-01, "nan", 1.533263972368564e+166, 1.156734948703097e+258]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_csk_loc4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.711553088491836e+01, 2.716314670820244e+01, 1.329133041243067e+01, 1.331643182991917e+01, -3.210845270819182e+00, -3.220434476431754e+00, 2.194194999410058e-01, -3.057136932447916e+00, "nan", "nan", -7.411761340967504e+150, -4.260319780195023e+150, 1.079010073129237e+243, 2.137755761021083e+243]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.179868818086794e-03, 0.000000000000000e+00, -3.171288841707055e-03, -8.852253049449785e-03, 0.000000000000000e+00, -8.846519510088510e-03, 4.153351655801080e+00, 0.000000000000000e+00, 4.158656121717748e+00, -4.298303616664001e+00, 0.000000000000000e+00, 7.829892461134373e+04, 4.179409719236300e+02, 0.000000000000000e+00, "nan", 6.733474112574475e+04, 0.000000000000000e+00, -1.135621965072012e+04, 8.236143114997760e+09, 0.000000000000000e+00, -3.782391318881807e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc4_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [2.171250000000000e-01, 2.171249999999999e-01, 2.150244236665798e-01, 2.150790398196276e-01, 2.471542939499245e-06, 2.424640990780594e-06, 2.171249999999999e-01, -1.925413468989174e-06, 9.395904503045099e-05, "nan", 1.779473038416902e-08, 2.171250000000002e-01, 0.000000000000000e+00, 2.171249999999999e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc4_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
