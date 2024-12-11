
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_g96_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.229913440133572e-01, -5.828244812739911e-01, -3.608408064309263e-01, -1.554114947358679e-01, -3.978816454180545e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_g96_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.279179944769183e-01, -1.154258973710232e-16, -7.262017915177634e-01, -2.333388220651943e-16, -4.114531430994776e-01, -2.029499126754557e-17, -7.051741185450337e-02, -3.685422106236010e-17, 2.570513114040454e-01, -4.810899229611381e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_g96_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.198402168950258e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.364138361236668e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.521501366406001e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.538671089253437e+01, 0.000000000000000e+00, 0.000000000000000e+00, -6.293221933490643e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
