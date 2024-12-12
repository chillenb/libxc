
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_gea2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.478889480628195e+01, -8.415383959640036e-01, 6.348963050106956e-01, 5.409116156005339e-01, 8.769545854576165e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_gea2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.388871680450201e+00, -1.890447141958958e-15, 2.673306887354114e+00, -1.075868378676951e-16, 8.769385073643793e-01, -4.032842788590949e-17, 1.000370885409721e-02, 2.590086599195396e-16, -7.596335982873273e-02, -1.721597645684904e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.658776870679791e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.462618591953954e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.229776970305012e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.695737829842652e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.621829333426592e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea2_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([1.666666666666667e-01, 0.000000000000000e+00, 1.666666666666667e-01, 0.000000000000000e+00, 1.666666666666668e-01, 0.000000000000000e+00, 1.666666666666666e-01, 0.000000000000000e+00, 1.666666666666668e-01, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
