
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vwn_3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.400974932567810e-02, -8.420806832234369e-02, -4.968925330417808e-02, -1.805934269911619e-02, -1.097300110097530e-02, -6.794969136077534e-03, -1.801842028956859e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vwn_3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.031839183693357e-01, -1.030206175508042e-01, -9.308584620330405e-02, -9.294613746644156e-02, -5.684163507694434e-02, -5.688441416110084e-02, -2.097191516243855e-02, -1.280949862758586e-01, -1.310880060288214e-02, -7.340401186536004e-02, -8.537666294694331e-03, -8.649255260185222e-03, -2.510897886364968e-04, -2.082304322818795e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
