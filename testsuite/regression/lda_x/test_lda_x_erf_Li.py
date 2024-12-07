
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_erf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_erf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.572793134389165e+00, -1.044606343128724e+00, -1.548517901998675e-01, -4.851124648902579e-02, -4.684712367910517e-03, -3.193188055498625e-05, -2.200286152831546e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_erf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_erf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.148499610198829e+00, -2.150580872455965e+00, -1.442884685893597e+00, -1.444247189752903e+00, -2.402963226461360e-01, -2.401422527867438e-01, -8.355579916216910e-02, -5.547110976930618e-05, -9.049445707783767e-03, -1.777764120876626e-09, -6.447151045380103e-05, -6.307964096559830e-05, -5.297763818454923e-10, -1.903419035021901e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
