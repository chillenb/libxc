
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc17_2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.224646737489428e-02, -3.838319304748419e-02, -6.584803996747660e-03, -6.791804614401876e-07, -2.167209554603299e-11, -3.906541101079743e-07, -1.707058909447101e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc17_2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([5.035061542102098e-04, 5.021475853014602e-04, 7.010660133066096e-04, 6.992789345659703e-04, -1.333479764304018e-02, -1.335182486099561e-02, -6.794367824029445e-07, -2.042148031345333e-03, -2.167210060271844e-11, -1.272062008917237e-04, -7.728470126472423e-07, -7.899582446445580e-07, -2.320382796193015e-12, -6.458300762839260e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
