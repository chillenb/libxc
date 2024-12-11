
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_sloc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_sloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.772386833374980e+00, -1.996330324333875e+00, -5.527779538696938e-01, -3.186249531970607e-01, -1.385973297006245e-01, -3.007978464178337e-02, -8.355492967724260e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_sloc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_sloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.602640241718451e+00, -3.605561578531754e+00, -2.594234567169866e+00, -2.596221740126945e+00, -7.187488037901407e-01, -7.184737007434355e-01, -4.143377756631697e-01, -3.749497605145996e-02, -1.801765590213648e-01, -1.679252897412914e-03, -3.923075830825352e-02, -3.897386906738934e-02, -1.167834519967179e-03, -8.590406586079237e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
