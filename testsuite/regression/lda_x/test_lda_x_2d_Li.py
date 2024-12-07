
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_2d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.834246404850115e+00, -2.218082225646937e+00, -2.609204580024534e-01, -1.041838486753562e-01, -2.601247963402268e-02, -2.038844358730323e-03, -5.240589488369279e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_2d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.747479508858421e+00, -5.755249209361907e+00, -3.324997641918054e+00, -3.329243616435856e+00, -3.915054662181197e-01, -3.912557484585228e-01, -1.563268183500093e-01, -2.851441106673856e-03, -3.901872609590331e-02, -1.610532336564060e-05, -3.074825282501590e-03, -3.041341174536455e-03, -8.791806462582125e-06, -5.269856683910135e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
