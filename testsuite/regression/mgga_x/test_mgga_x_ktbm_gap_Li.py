
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_gap_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.152805681681500e+00, -1.552347361470821e+00, -2.183679520480776e-01, -1.895297100811385e-01, -5.924277792597103e-02, -5.732789897353622e-03, -9.761325303281915e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_gap_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.498712348536138e+00, -2.501065261945713e+00, -1.730976493897700e+00, -1.731927203443437e+00, -3.315085732713999e-01, -3.376679961512082e-01, -2.309575574834392e-01, -7.537828289184107e-03, -9.849871603376917e-02, -2.388575199129130e-04, -6.561142123524442e-03, -7.869311398791540e-03, -1.318381812443700e-04, -1.134203705188512e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_gap_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.061666871160980e-04, 0.000000000000000e+00, -8.033753251867215e-04, -3.178403999064866e-03, 0.000000000000000e+00, -3.169191756002430e-03, -6.086357409629466e-02, 0.000000000000000e+00, -6.135514136588784e-02, -1.227295780760772e+01, 0.000000000000000e+00, -2.050575477996212e+01, -8.689926993469234e+01, 0.000000000000000e+00, -5.139511360469044e+04, -1.088709003224093e+00, 0.000000000000000e+00, -1.833378402036701e+01, -2.244358985749156e+00, 0.000000000000000e+00, -2.326741500228367e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_gap_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.420946762797282e-02, 1.416571263654944e-02, 3.880702201596078e-02, 3.870940877082747e-02, 3.043565207873772e-02, 3.195177023226707e-02, 9.652251335904209e-02, 2.627805313310022e-04, 4.958506384640277e-01, 2.094036935987672e-05, 3.307693549562020e-07, 2.673383677240175e-04, 2.157762924175931e-15, 1.015000802746931e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
