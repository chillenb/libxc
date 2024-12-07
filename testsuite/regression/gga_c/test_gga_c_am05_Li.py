
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_am05_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.798907692635378e-02, -7.616410670269010e-02, -4.087521110913900e-02, -1.737436253207036e-02, -9.176061734546917e-03, -5.489521929891202e-03, -1.361871606328820e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_am05_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.067342814650553e-01, -1.065817727084486e-01, -9.478299632050641e-02, -9.466081492503155e-02, -4.843963842498316e-02, -4.846729168569083e-02, -2.169399265288604e-02, -1.166921325583497e-01, -1.166005624402215e-02, -5.958851380899147e-02, -6.901556195753803e-03, -6.978954411214762e-03, -1.602101897804286e-04, -2.350859976567377e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_am05_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.735479682331374e-05, 0.000000000000000e+00, 1.729629997638306e-05, 5.577640445739186e-05, 0.000000000000000e+00, 5.560144906152712e-05, 8.514181343982222e-04, 0.000000000000000e+00, 8.472057187610143e-04, 7.318098836844033e-01, 0.000000000000000e+00, 1.066250226892355e-02, 1.504328181589803e+00, 0.000000000000000e+00, 1.300163942584289e+00, 3.862911898188336e-03, 0.000000000000000e+00, 3.633273469950266e-03, 2.174451917539076e-02, 0.000000000000000e+00, 4.378201885884758e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
