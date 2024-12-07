
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.861664941891038e-01, -6.229106159176491e-01, -3.738229093706273e-01, -1.319408327148993e-01, 2.041512546527145e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.211734381004238e-01, 7.375047491137587e-02, -7.835098042666019e-01, 1.251377202583918e-01, -4.296070677123447e-01, 9.837183618164036e-02, -1.052194367463868e-01, 1.682652138216760e-01, 1.125083463052247e-02, 1.857795951771590e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.268373819686489e-02, -3.211159324715696e-02, 7.604546815524012e+17, -2.347224383972815e-02, -4.823431369716116e-02, 6.213489604372652e+17, -1.756569167556403e-01, -3.798275443256743e-01, 5.550336171511199e+15, -8.536090319229103e+00, -3.836444662563308e+01, -1.059896910600514e+18, -1.667601804229714e+03, -3.457271338190035e+04, -1.953311237199593e+18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
