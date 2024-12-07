
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ow_lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ow_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.282228851632633e-02, -4.051591494652124e-02, -2.601129955912549e-02, -2.131252863485033e-05, -5.388673395773836e-09, -2.081842897120096e-03, -3.003770849422545e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ow_lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ow_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.478339646982935e-02, -4.455199620284490e-02, -4.299868117141432e-02, -4.279185999270867e-02, -3.006276769604229e-02, -3.012915305404469e-02, -4.796393382254552e-06, -6.406275543256243e-02, -1.507422598674778e-09, -3.162926915632686e-02, -2.701316791441482e-03, -2.792504526249012e-03, -2.079686215637139e-05, -9.360839371319269e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
