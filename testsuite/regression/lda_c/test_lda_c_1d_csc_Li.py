
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_1d_csc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_1d_csc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.912505747455010e-05, -4.989343518712614e-04, -6.196506730483305e-02, -1.318882691160258e-03, -9.548311964528598e-05, -2.356079010965352e-05, -2.178453324631966e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_1d_csc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_1d_csc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [5.730367062398530e-05, 5.758212150025507e-05, 4.554390657952189e-04, 4.576416008045332e-04, -8.608321530421560e-02, -8.621955579372864e-02, -2.478890496272364e-03, -4.477743698048208e-02, -1.793907646887854e-04, -4.549898392887188e-03, -4.461335694641442e-05, -4.554941625206079e-05, -3.064713282409270e-10, -7.602444941821183e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
