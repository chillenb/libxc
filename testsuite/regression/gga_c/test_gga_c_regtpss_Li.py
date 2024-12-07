
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_regtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.038653647892862e-02, -4.405609879116904e-02, -3.600994651649092e-03, -1.562523635986147e-02, -2.072351503042441e-03, -2.003205894392235e-08, -5.655198531684391e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_regtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.160054473225238e-01, -1.158790514191272e-01, -1.016667999123736e-01, -1.015714119411483e-01, -1.809593130275794e-02, -1.810245897555880e-02, -2.371893347324360e-02, -1.022646629006360e-01, -9.074184510240928e-03, 4.505349605283845e-01, -1.285890271154643e-07, -1.292401623948396e-07, -3.577738420194182e-15, -4.233382578950479e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_regtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.567902691829763e-05, 9.135805383659526e-05, 4.567902691829763e-05, 1.483917510803814e-04, 2.967835021607628e-04, 1.483917510803814e-04, 3.560920424366704e-03, 7.121840848733407e-03, 3.560920424366704e-03, 2.900691718245773e+00, 5.801383436491545e+00, 2.900691718245773e+00, 1.528420097047116e+01, 3.056840194094232e+01, 1.528420097047116e+01, 4.417232085468924e-04, 8.834464171198456e-04, 4.417232085468924e-04, 5.059265615793612e-06, 1.011868011739185e-05, 5.059265615793612e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
