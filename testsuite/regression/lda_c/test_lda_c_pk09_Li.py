
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pk09_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pk09", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.360567008130327e-02, -8.375599716065790e-02, -4.965273754784837e-02, -1.911972812357798e-02, -1.114256555017351e-02, -6.567230452403835e-03, -1.413018225406602e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pk09_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pk09", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.028511398853998e-01, -1.026884235812027e-01, -9.264200854922940e-02, -9.250347248812096e-02, -5.669544321199237e-02, -5.673772580439447e-02, -2.195319727393639e-02, -8.986603629109918e-01, -1.322369947742898e-02, -1.028780914900781e+02, -8.389862670080760e-03, -8.471798406256123e-03, -1.879828401535707e-04, -1.697830182028841e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
