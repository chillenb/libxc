
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc17_2_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.224646691606159e-02, -3.838319247732855e-02, -6.584806712772819e-03, -5.118847965201458e-04, -3.180640280010177e-05, -3.907009480391759e-07, -2.194670889750603e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc17_2_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [5.028256536466076e-04, 7.001759310406998e-04, -1.334331135294370e-02, -1.025010716081849e-03, -6.361765751515152e-05, -7.814026287776596e-07, -4.389341779524326e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
