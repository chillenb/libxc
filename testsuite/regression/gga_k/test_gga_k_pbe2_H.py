
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pbe2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.046486285970077e+00, 2.179882158457330e+00, 1.084886205542248e+00, 2.305656936124474e-01, 6.993878601454426e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pbe2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.377560344493749e+00, 5.101058403232819e-16, 2.250949414438000e+00, 5.380024783069018e-16, 5.975956580640334e-01, 8.394377304592335e-17, 2.278063993892241e-01, 9.484341404753808e-17, 1.164454412685513e-03, -9.722164545549962e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pbe2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.100137086529538e-01, 0.000000000000000e+00, 0.000000000000000e+00, 6.420121711539043e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.643756469612511e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.761222852761161e+01, 0.000000000000000e+00, 0.000000000000000e+00, 9.525288199560393e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
