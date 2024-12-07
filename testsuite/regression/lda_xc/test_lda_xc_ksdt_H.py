
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_ksdt_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_ksdt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.531440416487824e-01, -5.877063939891897e-01, -3.507852528235211e-01, -1.003884386068968e-01, -5.562691142387084e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_ksdt_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_ksdt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.640227283820993e-01, -3.216458804179927e-01, -7.771032518289921e-01, -3.068188498186428e-01, -4.627537835336134e-01, -2.360565699536945e-01, -1.318426038110058e-01, -1.012468034557128e-01, -7.350617331440476e-03, -7.545856746814281e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
