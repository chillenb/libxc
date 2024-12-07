
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vwn_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.400974967430613e-02, -8.420806866583493e-02, -4.968925336996970e-02, -1.805962574706621e-02, -1.097300126798731e-02, -6.795015886534831e-03, -1.629259198827822e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vwn_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.031834018478338e-01, -1.030211332894431e-01, -9.308530752378019e-02, -9.294667541839198e-02, -5.684184125280023e-02, -5.688420787823826e-02, -2.097197346039268e-02, -1.289435150906530e-01, -1.310880055262679e-02, -7.438430131514800e-02, -8.546148638049592e-03, -8.640658139149656e-03, -1.914214508247163e-04, -2.832624807333714e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
