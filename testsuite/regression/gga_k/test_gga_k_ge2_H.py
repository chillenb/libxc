
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ge2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035145370420630e+00, 1.687464332401589e+00, 6.148959394071615e-01, 9.639340498542932e-02, 7.619949676053718e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ge2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388871680450207e+00, 3.496021547817053e-16, 2.673306887354115e+00, 4.650591113897583e-16, 8.769385073643794e-01, 1.152933514152392e-16, 1.000370885409723e-02, 3.943193004512873e-17, -7.596335982873284e-02, -1.500131900024939e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ge2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.658776870679791e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.462618591953954e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.229776970305012e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.695737829842652e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.621829333426592e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
