
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_l_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([3.728182957207354e-11, 4.841566230577192e-12, -1.259018978182929e-11, 5.508695706149481e-11, 7.523435542645333e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_l_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.466804359450984e-02, -1.897930891974544e-01, 1.079414211491272e-02, -1.925295191270592e-01, -2.404755157791287e-02, -1.736102302375463e-01, 4.800427159893710e-02, 4.129820081659515e-02, 1.291745762211433e-02, 6.415863633662444e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_l_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.009001734956082e+00, 0.000000000000000e+00, 5.179591447286795e+16, -1.337005391151593e-02, 0.000000000000000e+00, -4.696296100166442e+16, 1.400486813413244e-01, 0.000000000000000e+00, -1.196753222310857e+17, -1.440895635610213e+01, 0.000000000000000e+00, 3.026683219178990e+16, -2.752497663256910e+04, 0.000000000000000e+00, 7.813734192281302e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_l_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.406453599545652e+00, -7.726969779634408e+05, 2.298699084332589e-02, -2.978827608011846e+05, -4.817968776328787e-02, 2.566773775058609e+05, 9.441289405075137e-02, 1.802561905188958e+05, 1.885729520012959e-02, 1.072219428370901e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
