
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_edmggah_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.882933648269891e-01, -3.861472779304497e-01, -2.846860935149308e-01, -1.334155152285138e-01, -5.440656521916028e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_edmggah_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.300737220471338e-01, -1.292146032167649e-01, -5.949856743299970e-01, -1.242980638337788e-01, -3.519316369787120e-01, -9.854944047043196e-02, -1.010896537408962e-01, -4.158882017096328e-02, -2.484611140493635e-02, -2.453540189748340e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.231374198417420e-04, 1.472661555066064e-16, 7.363307775330318e-17, -7.962196442344985e-03, 3.240069935630985e-16, 1.620034967815492e-16, -1.463651377419983e-01, 1.389767707807788e-14, 6.948838539038938e-15, -7.236066323075533e+00, 3.118292098341930e-11, 1.559146049170965e-11, -2.591076236026971e+04, 1.253737775995283e-25, 6.268688879976413e-26]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-7.342021263181333e-05, -2.169436830976663e-17, -3.422329822679595e-03, -3.424881429235486e-17, -1.258817035353157e-02, -2.977325616464631e-16, -1.185335603008501e-02, -1.276980726713362e-14, -4.437849187101031e-03, -5.368328357205251e-33]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_edmggah_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_edmggah", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.936808505270777e-04, -5.780540928528317e-31, 1.368931929071810e-02, -1.262390896823658e-30, 5.035268141412391e-02, -5.526788394415427e-29, 4.741342412023787e-02, -1.247265140208141e-25, 1.775139674840412e-02, -5.014951166539975e-40]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
