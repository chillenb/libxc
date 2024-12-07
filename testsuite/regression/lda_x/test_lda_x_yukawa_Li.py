
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_yukawa_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_yukawa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.599090515080575e+00, -1.072459039100240e+00, -1.855466281829784e-01, -7.287521908398913e-02, -1.201774212846440e-02, -1.244781577550382e-04, -8.801054394122955e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_yukawa_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_yukawa", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.173433238076705e+00, -2.175511162887394e+00, -1.469282541233892e+00, -1.470641384182910e+00, -2.724577440154656e-01, -2.723041623365125e-01, -1.134361956060481e-01, -2.150687327259381e-04, -2.105393935819517e-02, -7.110825595322782e-09, -2.491707698832553e-04, -2.439085805635886e-04, -2.119074828713793e-09, -7.613620396633974e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
