
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_lda_xc_bn05_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_bn05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.447307348688094e+00, -9.353045966852145e-01, -1.218187305366437e-01, -2.546760464086229e-02, -2.209399101433014e-03, -2.631825561474588e-05, -1.929964819468595e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_lda_xc_bn05_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_bn05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.010316582350347e+00, -2.012163826734952e+00, -1.317047184152542e+00, -1.318192512951431e+00, -1.939887666627391e-01, -1.939136674420968e-01, -4.592119289213315e-02, -3.483742981894173e-02, -4.275669255675832e-03, -3.374427939185533e-03, -5.182838650380531e-05, -5.153502553628217e-05, -4.004497835291321e-10, -3.408129989298175e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
