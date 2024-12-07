
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_apf_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.114047517646871e-01, -4.646088510188204e-01, -2.861831357327527e-01, -1.069295391750714e-01, -2.113197476009185e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_apf_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.751544619432718e-01, 1.482753301857709e+00, -5.950386553551745e-01, 6.903947922619511e+01, -3.393421722562173e-01, 4.038254263339245e+01, -9.785630589447647e-02, 3.417187736013451e-01, -9.179646187771612e-03, -5.521863293240203e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_apf_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.963479231413809e-03, 3.118474191588641e-02, 1.559237095794320e-02, -8.955442158548188e-03, 1.818037064282878e-02, 9.090185321414391e-03, -8.645507297245672e-02, 7.700105081830898e-02, 3.850052540915448e-02, -5.010452734510430e+00, 1.707459878037234e-01, 8.537299390186143e-02, -1.517384334774384e+04, 1.839477951744348e-03, 9.197389761389030e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
