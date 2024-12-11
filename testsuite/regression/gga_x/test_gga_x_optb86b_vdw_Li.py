
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optb86b_vdw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb86b_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.769799420988983e+00, -1.251759478057900e+00, -4.000634901614446e-01, -1.586960762622399e-01, -7.568609055561799e-02, -4.774392146738944e-02, -3.045749459589800e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optb86b_vdw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb86b_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.270310220340050e+00, -2.272431446507987e+00, -1.548376516055549e+00, -1.549756201097790e+00, -3.211562196666036e-01, -3.210472408119966e-01, -2.069311765116041e-01, -4.252606205080187e-02, -7.093901658124452e-02, -3.424432200025037e-03, -4.405340262434056e-02, -4.402868978183523e-02, -2.709311582650223e-03, -2.106849315490249e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optb86b_vdw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optb86b_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.514799848144694e-04, 0.000000000000000e+00, -1.509437475960485e-04, -6.319902618110907e-04, 0.000000000000000e+00, -6.298881382883187e-04, -1.029168311351457e-01, 0.000000000000000e+00, -1.028813748497893e-01, -2.293509061201834e+00, 0.000000000000000e+00, -1.827523371870675e+02, -6.559400120477999e+01, 0.000000000000000e+00, -1.503113185396764e+06, -1.632024104603690e+02, 0.000000000000000e+00, -1.615173198793642e+02, -3.487023502010739e+06, 0.000000000000000e+00, -9.128600620887235e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
