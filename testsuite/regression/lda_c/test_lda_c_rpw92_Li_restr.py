
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_rpw92_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rpw92", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.348350413039079e-02, -8.371486687369814e-02, -4.959806866989800e-02, -3.330689993222388e-02, -1.948850161575657e-02, -6.778913305372585e-03, -1.789530721783893e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_rpw92_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_rpw92", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.025617742488498e-01, -9.247599975071723e-02, -5.666713531650919e-02, -3.908979809355455e-02, -2.363143054234018e-02, -8.569322884269133e-03, -2.365942116388912e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
