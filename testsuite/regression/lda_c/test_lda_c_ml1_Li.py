
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ml1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ml1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.589211906555165e-02, -8.081740633039308e-02, -5.067145429236869e-02, -3.544504619910358e-03, -1.554700178031944e-04, -4.273102537158570e-03, -7.283390082466559e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ml1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ml1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.004768634688148e-02, -8.992923115125193e-02, -8.606211709694392e-02, -8.595149850033028e-02, -5.878252011459952e-02, -5.882693577099556e-02, -3.595579237299666e-03, -3.106103462127434e+00, -1.604639811141824e-04, -2.734321286695725e+02, -5.563987747432398e-03, -5.650460126295149e-03, -7.511391685914256e-05, -1.581365472649518e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
