
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lc94_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lc94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.035261023468491e+00, 1.699817356189841e+00, 6.196142519237090e-01, 7.139318391584340e-02, 8.791004056685343e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lc94_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lc94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.388489133634319e+00, 1.801261530821365e-16, 2.675823488757672e+00, 3.111103467559230e-16, 8.960402475472506e-01, 1.808122158602878e-16, 6.521924807363143e-02, 4.404408924080950e-17, 3.701664644040117e-04, -2.576487795192348e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lc94_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lc94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.541215028558721e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.302034168323425e-02, 0.000000000000000e+00, 0.000000000000000e+00, 2.984349004215619e-01, 0.000000000000000e+00, 0.000000000000000e+00, 6.052280238138692e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.787136291699757e+02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
