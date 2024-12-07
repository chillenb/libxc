
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vwn_4_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.240553225520315e-02, -3.111652546340106e-02, -2.512969361463155e-02, -1.327316222651283e-02, -1.560337027758898e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vwn_4_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.637848240923835e-02, -2.753551741435706e-01, -3.502021717295450e-02, -2.630613468310200e-01, -2.865862581079151e-02, -2.064254686744887e-01, -1.570221632891375e-02, -9.827352787236474e-02, -1.993820725752317e-03, -7.563552419899674e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
