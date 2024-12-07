
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_br78_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_br78", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.282804458993565e-02, -4.358239107160200e-02, -1.549873371037501e-02, -7.305036256840228e-03, -3.026485679651004e-03, -7.146679584018696e-04, -1.279225889848908e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_br78_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_br78", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.131146928128919e-02, -5.189873562847196e-02, -1.987948913903588e-02, -9.565549518539673e-03, -4.005362296623609e-03, -9.512204577516869e-04, -1.705581008901258e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
