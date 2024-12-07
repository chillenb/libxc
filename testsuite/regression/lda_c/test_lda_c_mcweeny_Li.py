
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_mcweeny_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_mcweeny", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.169892054581030e-02, -8.728200864536331e-02, -5.823849663015496e-02, -4.904579072594838e-05, -1.268774258592265e-08, -4.984515051548318e-03, -7.234725334359089e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_mcweeny_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_mcweeny", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.545970586873825e-02, -9.496418927138507e-02, -9.208890487486791e-02, -9.164335729358640e-02, -6.666475260837831e-02, -6.681338737533897e-02, -1.054566965334253e-05, -1.474249284111533e-01, -3.469252113224665e-09, -7.447176611593211e-02, -6.458105339639746e-03, -6.676434319839691e-03, -5.008752061624984e-05, -2.254575665999139e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
