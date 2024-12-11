
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_teter93_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_teter93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.539759179016904e-01, -5.884513347580023e-01, -3.512072474610619e-01, -1.003633099985461e-01, -5.652168091563119e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_teter93_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_teter93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.651331793364321e-01, -2.638312388691184e-01, -7.781010173845341e-01, -2.508778840288693e-01, -4.633362301509812e-01, -1.938721613999246e-01, -1.318686686489499e-01, -9.079424252206676e-02, -7.463077467591797e-03, -7.254251597688185e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
