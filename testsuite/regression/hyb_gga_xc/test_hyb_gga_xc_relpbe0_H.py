
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_relpbe0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_relpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.975998418859518e-01, -4.453952269407466e-01, -2.646484012724993e-01, -8.591694076795731e-02, -6.662032786874668e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_relpbe0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_relpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.595615505532539e-01, 5.792079939703445e-01, -5.911335960073658e-01, 2.911589815148399e+01, -3.442822082320379e-01, 1.610371840257471e+01, -8.181780556753787e-02, 1.082742093560162e-01, -8.818208669179557e-03, -1.548348584901657e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_relpbe0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_relpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.107276540961146e-03, 1.306303482796397e-02, 6.531517413981984e-03, 3.066497877777060e-04, 8.012163479824180e-03, 4.006081739912090e-03, -1.435003849225739e-02, 3.270615353699301e-02, 1.635307676849651e-02, -3.633998645533486e+00, 6.670042982089497e-02, 3.335021491044746e-02, -3.555144958842460e+01, 7.484747126777127e-04, 3.742373563982494e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
