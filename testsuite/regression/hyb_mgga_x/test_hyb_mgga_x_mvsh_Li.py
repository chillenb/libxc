
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_mvsh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.528546771917837e+00, -1.060946485304757e+00, -2.400954958596514e-01, -1.379865189068875e-01, -5.411174945261393e-02, -1.934214177036678e-03, -5.275517387994058e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_mvsh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.977544165109858e+00, -1.979410541810801e+00, -1.331590108305341e+00, -1.332682834347391e+00, -1.801090437830499e-01, -1.109875922813131e-01, -1.811937989584108e-01, 1.509171296198761e+00, -4.733587072438472e-02, 3.901455530759419e+00, -3.341574169913280e-03, 1.506043642310819e+00, -1.155297369646392e-05, 1.146475556315176e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mvsh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.742083375432182e-04, 0.000000000000000e+00, -2.732855899563823e-04, -1.174164090305753e-03, 0.000000000000000e+00, -1.171579110508994e-03, -2.269745326168519e-01, 0.000000000000000e+00, -3.159459527151910e-01, -4.052863783210071e+00, 0.000000000000000e+00, -3.874636942400716e+04, -1.571570681766683e+02, 0.000000000000000e+00, -7.912318946899151e+09, 9.275724882251772e+00, 0.000000000000000e+00, -3.311701939394402e+04, 8.385221554390977e+03, 0.000000000000000e+00, -2.216750282141541e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mvsh_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mvsh_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.428211945938872e-02, 1.427245966606878e-02, 2.054639113129936e-02, 2.055319830260231e-02, 6.032712404031915e-02, 8.181482893226001e-02, 1.559238152217558e-01, 4.950270338032584e-01, 3.935800296341502e-01, 3.223761985970341e+00, 7.933200232909227e-12, 4.813487182055661e-01, 7.317198188019058e-24, 9.670181514041916e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
