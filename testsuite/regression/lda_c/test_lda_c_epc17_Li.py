
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc17_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.667310831389436e-02, -8.870742267589361e-02, -6.593683903974057e-03, -6.791804689683788e-07, -2.167209554603346e-11, -3.906541101098799e-07, -1.707058909447101e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc17_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.388528240932703e-03, 2.382083473219949e-03, 3.744524148876661e-03, 3.734979028483981e-03, -1.337078705376020e-02, -1.338786022663424e-02, -6.794367974650088e-07, -2.042148076616602e-03, -2.167210060271939e-11, -1.272062008917293e-04, -7.728470126547823e-07, -7.899582446522649e-07, -2.320382796193015e-12, -6.458300762839260e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
