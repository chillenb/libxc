
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_revapbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.645264609327986e+01, 8.201242892226375e+00, 6.913186935936576e-01, 1.324477729144583e-01, 2.768810126732575e-02, 1.532845269043814e-03, 5.452312965178888e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_revapbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.587960714775697e+01, 2.592713869425544e+01, 1.226788964145916e+01, 1.228920415637111e+01, 7.890665961790164e-01, 7.891438154638981e-01, 2.134116636764148e-01, 2.323558101848670e-03, 3.226908605716674e-02, 2.343131834516740e-06, 2.568629247958795e-03, 2.531743792648414e-03, 1.045386035766431e-06, 5.283293025421091e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_revapbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_revapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.600571795436173e-03, 0.000000000000000e+00, 2.593835717109334e-03, 7.321464705585723e-03, 0.000000000000000e+00, 7.303797406180969e-03, 1.761685087651594e-01, 0.000000000000000e+00, 1.757774170511295e-01, 3.604254684805192e+00, 0.000000000000000e+00, 3.499330242291133e-02, 3.036776391465928e+01, 0.000000000000000e+00, 7.106597602428821e-03, 3.738993748089071e-02, 0.000000000000000e+00, 3.466359637349536e-02, 3.455500357238084e-03, 0.000000000000000e+00, 3.516292908779982e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
