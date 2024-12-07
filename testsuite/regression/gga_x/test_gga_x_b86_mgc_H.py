
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b86_mgc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220826467052728e-01, -5.796002940594310e-01, -3.622771098825087e-01, -1.382711752604861e-01, -1.486783173395086e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b86_mgc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282184742528140e-01, -8.339915584609322e-17, -7.167725126079185e-01, -1.498147256425273e-16, -4.006673137712776e-01, 5.326707684445923e-17, -1.313612851539929e-01, -6.908041352767684e-17, -1.407013770906176e-02, -8.438892678763208e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b86_mgc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.879074291160656e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.602438534548128e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.798880599846760e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.965709260642278e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.597607004337428e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
