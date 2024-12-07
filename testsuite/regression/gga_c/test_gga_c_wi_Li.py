
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_wi_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.259374702740577e-01, -8.861249385038167e-02, 1.566069915934520e-03, -1.526579017861598e-02, 1.666183621919468e-04, -1.679910921105222e-09, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_wi_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.852149045300704e-01, -1.852149045300704e-01, -1.755705107188749e-01, -1.755705107188749e-01, -1.354502690619071e-03, -1.354502690619071e-03, -2.490907776579140e-02, -2.490907776579140e-02, -1.312968726539917e-03, -1.312968726539917e-03, -1.007945735951997e-08, -1.007945735951997e-08, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_wi_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.038667809226341e-05, 6.077335618452683e-05, 3.038667809226341e-05, 1.765246169358687e-04, 3.530492338717374e-04, 1.765246169358687e-04, 8.330039095665344e-04, 1.666007819133069e-03, 8.330039095665344e-04, 2.327472157561142e+00, 4.654944315122284e+00, 2.327472157561142e+00, 3.358901585679285e+00, 6.717803171358569e+00, 3.358901585679285e+00, 3.244243214369047e-05, 6.488486428738093e-05, 3.244243214369047e-05, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
