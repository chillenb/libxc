
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tw2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.643434536728423e+01, 8.171980788386026e+00, 6.214709948347235e-01, 1.323865660578513e-01, 2.624064775421799e-02, 1.146032152825097e-03, 4.073814995268807e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tw2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.593161785636838e+01, 2.597907426632732e+01, 1.235282287805953e+01, 1.237411442761510e+01, 8.455652651476050e-01, 8.454548942265464e-01, 2.135730344504539e-01, 1.738753602505419e-03, 3.474032969133899e-02, 1.750728125629447e-06, 1.922481067797155e-03, 1.894720450279430e-03, 7.810838018979924e-07, 3.947529471114987e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tw2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.459145779393316e-03, 0.000000000000000e+00, 2.452983331193492e-03, 6.617032501738280e-03, 0.000000000000000e+00, 6.601731765731282e-03, 9.237176732949995e-02, 0.000000000000000e+00, 9.206181269677673e-02, 3.476176264113228e+00, 0.000000000000000e+00, 1.045433108545895e-02, 1.968116443965221e+01, 0.000000000000000e+00, 2.119735079601433e-03, 1.117237685318557e-02, 0.000000000000000e+00, 1.035684872008194e-02, 1.030693914396762e-03, 0.000000000000000e+00, 1.048826251654453e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
