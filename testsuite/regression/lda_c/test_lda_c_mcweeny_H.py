
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_mcweeny_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_mcweeny", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([5.418922766590002e-18, 5.097279038609098e-18, -3.443224332421142e-18, -9.569185202909929e-19, -4.204435468728985e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_mcweeny_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_mcweeny", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.481538703457384e-17, -2.844554576493826e-01, 3.187202734936021e-17, -2.745314789840047e-01, -3.977593405733886e-18, -2.214740377818834e-01, -1.201054007851296e-18, -9.723159010484733e-02, -5.263856084411332e-19, -5.899580403606445e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
