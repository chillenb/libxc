
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ol2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.267901831548777e-01, -5.815452939714855e-01, -3.618484885930042e-01, -1.499959742906981e-01, -1.541403329366045e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ol2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.612843312268491e-01, -3.110698023773574e-17, -6.931593550479307e-01, -2.376121731607671e-16, -4.228050428032224e-01, -1.150612883636217e-17, -5.499968469381717e-02, -7.346807395587777e-17, 2.042960557027783e+00, -1.153651985040133e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ol2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.186909557650576e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.819703619260165e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.302926228695699e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.632062456735581e+01, 0.000000000000000e+00, 0.000000000000000e+00, -3.274754451456279e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
