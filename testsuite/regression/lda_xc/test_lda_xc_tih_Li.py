
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_tih_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_tih", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=False, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.915290854273275e+00, -2.915290854273275e+00, -1.770587138438609e+00, -1.770587138438609e+00, -5.752631244872439e-01, -5.752631244872439e-01, -4.056630878606430e-01, -4.056630878606430e-01, -3.902275060777994e-01, -3.902275060777994e-01, -3.892066653463063e-01, -3.892066653463063e-01, -3.891939614134251e-01, -3.891939614134251e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
