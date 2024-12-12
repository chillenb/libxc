
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbrxh_bg_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.720472532085499e+00, -1.366637443034971e+00, -9.870454496753268e-01, -1.523946021804781e-01, -1.411100810281256e-01, -6.727185683065568e+00, -8.893744457891069e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbrxh_bg_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.857444831392968e+00, -1.860326266487682e+00, -9.023505185366951e-01, -9.031376918682585e-01, 2.743102362739376e-01, 2.770519846189390e-01, -1.855575344774963e-01, 7.175649620374538e+00, -9.072018571444811e-04, 5.996070563431758e+01, 6.801948382676692e+00, 7.089268974172984e+00, 9.901101184804547e+01, 1.141996061533276e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxh_bg_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.269279927351925e-04, 0.000000000000000e+00, -6.233684315204610e-04, -4.310400905713349e-03, 0.000000000000000e+00, -4.297757629171459e-03, -7.610594443548530e-01, 0.000000000000000e+00, -7.623649958064457e-01, -5.658952554148619e+00, 0.000000000000000e+00, -1.574741330882202e+05, -3.995957179266806e+02, 0.000000000000000e+00, -9.823028639763770e+10, -1.299217756666106e+05, 0.000000000000000e+00, -1.333737039764843e+05, -4.637575438150430e+11, 0.000000000000000e+00, -1.770732115916870e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxh_bg_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-9.934029536375221e-03, -9.914030664705240e-03, -1.479812764457424e-02, -1.480456891846341e-02, -3.697252415372498e-03, -3.685156842724763e-03, -8.549510886131984e-02, -6.207624491035486e-05, -3.865516866153639e-02, -3.138844914940991e-06, -6.639523730733190e-05, -6.350077341847037e-05, -1.680989021419393e-06, -1.394362117866725e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
