
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.078815075533197e-02, -5.529339683609313e-02, 1.880822776915780e-02, 1.048412757691714e-03, -7.716253397523075e-09, -1.114119606316758e-02, -5.379280737425244e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.650398687510164e-02, -9.589869866635181e-02, -7.798284636664432e-02, -7.751547216229604e-02, -1.285924233452381e-01, -1.329557493177933e-01, 2.234185544391546e-02, -1.799444538072786e-02, -3.333085183183028e-02, -8.119806915376707e-03, -2.774702026131097e-02, -2.260425325471887e-02, -9.564571090430467e-04, -5.400723578196803e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.612637109969793e-04, 0.000000000000000e+00, -1.614835015153556e-04, -3.278046777525293e-04, 0.000000000000000e+00, -3.295230859191104e-04, 1.308261127616430e-01, 0.000000000000000e+00, 1.404647161337976e-01, -2.692265161507612e+01, 0.000000000000000e+00, 3.448820627447375e+02, 1.944951263716050e+02, 0.000000000000000e+00, 2.125516494923458e+06, 1.386661775616248e+01, 0.000000000000000e+00, 4.748855933292826e+02, 3.798904266002201e+01, 0.000000000000000e+00, 1.010086647635986e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.450119352844756e-02, 1.451485963112269e-02, 1.157207940925500e-02, 1.160909832699235e-02, -2.209451558999569e-02, -2.490814280156245e-02, 9.733068674329906e-01, -5.317320620634892e-03, -4.651304600391167e-01, -8.694666503706216e-04, -3.553810802052805e-06, -6.835424367518390e-03, -3.197146230413648e-14, -4.406275367863556e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
