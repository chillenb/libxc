
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_zlp_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_zlp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.004644391068613e+00, -1.422986487107128e+00, -3.576695431177697e-01, -1.556344502574414e-01, -6.204111703825632e-02, -1.436419585019110e-02, -2.556640451208206e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_zlp_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_zlp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.626524615291618e+00, -1.872223910283283e+00, -4.748121501320156e-01, -2.070567266987307e-01, -8.263771004128437e-02, -1.914677353702482e-02, -3.408827671355943e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
