
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_tih_H_restr_1_vrho():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_tih", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=False, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.454452284862438e-01, -8.660590950848160e-01, -5.267815480696112e-01, -3.920232734274612e-01, -3.891942575755160e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
