
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc17_2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.254032475539053e-16, -4.238187677723340e-16, -4.250186364888066e-16, -4.255303859295150e-16, -4.255319113935911e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc17_2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.225206422928001e-16, -1.268608865587042e-01, -4.199600402259781e-16, -9.145156151490554e-02, -4.237011503490664e-16, -1.829898967733249e-02, -4.255645274472308e-16, -3.485306153354979e-04, -4.255319152596873e-16, -3.644135275461070e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
