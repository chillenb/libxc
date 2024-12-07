
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lieb_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lieb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035912109612808e+00, 1.722587724908679e+00, 6.522293213585163e-01, 1.344245175516000e-01, 1.274360697141224e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lieb_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lieb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388104941258029e+00, 2.722319124247368e-16, 2.638183494847025e+00, 3.410214700334765e-16, 8.396051254130238e-01, 7.179057864264180e-17, -2.802740371211987e-02, 5.473346200038179e-17, -1.271999333806162e-01, -1.441727862009799e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lieb_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lieb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.794984951698324e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.081314174754547e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.404007012938523e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.837279232847288e+01, 0.000000000000000e+00, 0.000000000000000e+00, 2.713616813856664e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
