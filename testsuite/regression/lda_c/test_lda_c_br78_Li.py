
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_br78_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_br78", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.282794817797101e-02, -4.358232009284781e-02, -1.549872740085412e-02, -9.715278707825408e-06, -2.062486869898279e-09, -7.145822828040722e-04, -9.950074804668515e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_br78_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_br78", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.145428430725824e-02, -6.116881611449777e-02, -5.201003017606230e-02, -5.178755595021235e-02, -1.985971594192467e-02, -1.989927138838549e-02, -3.009585161719060e-06, -2.920371759042606e-02, -6.670843501389517e-10, -1.210593926071393e-02, -9.356279054455541e-04, -9.669276446298304e-04, -6.891206815591305e-06, -3.101022993521878e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
