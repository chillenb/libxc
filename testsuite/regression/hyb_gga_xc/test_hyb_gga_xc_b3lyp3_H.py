
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3lyp3_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.038248303906421e-01, -4.682448858648194e-01, -2.910915530125324e-01, -1.153492591911472e-01, -4.346418835095610e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3lyp3_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.694877618999318e-01, -2.422740255227088e-01, -5.836836142475654e-01, -2.510736540782595e-01, -3.315719059344060e-01, -1.972652439477766e-01, -8.720234145936633e-02, -4.651984540065803e-02, -1.174084263108945e-02, -1.618595547035828e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3lyp3_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.505015750542190e-02, 2.369104295041123e-02, 1.776147695649069e-02, -1.830735392729338e-02, 3.764003296978145e-02, 2.819338713432885e-02, -1.214902915585818e-01, 3.229284926813911e-01, 2.421892139902259e-01, -7.453445188105283e+00, 1.379257500640840e+01, 1.034441375290050e+01, -3.691337851912702e+04, 5.797825996786096e-18, 4.348362994013478e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
