
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.930394000418183e+00, -1.278801346889639e+00, -2.857134329083767e-01, -1.779403805671163e-01, -5.909470330257174e-02, -1.253092272732911e-02, -2.325999257782303e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.732782619976827e+00, -2.735444859866205e+00, -1.884080184583410e+00, -1.885824076492866e+00, -3.542456999368007e-01, -3.534498980377455e-01, -2.467886511528513e-01, -1.518163660916648e-02, -7.542498887234228e-02, -4.814961149452113e-04, -1.596467316451031e-02, -1.584840248537442e-02, -3.216117530658952e-04, -2.331593387195087e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.935662167332068e-04, 0.000000000000000e+00, -5.915502807001307e-04, -2.237175154739451e-03, 0.000000000000000e+00, -2.231111561192556e-03, -4.404249233738881e-02, 0.000000000000000e+00, -4.609399684465914e-02, -9.287029952412961e+00, 0.000000000000000e+00, -1.961391763937066e+01, -7.100520214860478e+01, 0.000000000000000e+00, -4.911805415042815e+04, -3.646828035667346e-01, 0.000000000000000e+00, -1.753732954169499e+01, -7.437873750247501e-01, 0.000000000000000e+00, 2.911748634548703e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.283657302723323e-02, 2.282034286964581e-02, 2.630236559983822e-02, 2.631361136985142e-02, -4.914146711364954e-03, -5.059261349816589e-03, 2.691650561790458e-01, 2.507221489583975e-04, -2.819984259021104e-02, 2.001247934708891e-05, 5.416075768957198e-06, 2.550448890625306e-04, 9.030772834378464e-11, -1.268198121621011e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
