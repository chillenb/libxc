
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_c09x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_c09x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.761406710332421e+00, -1.240527210451837e+00, -3.887834756301400e-01, -1.582387555035121e-01, -7.338218840821149e-02, -2.558213622412285e-02, -4.776957878960282e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_c09x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_c09x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.280508411636826e+00, -2.282622872063324e+00, -1.560668082940854e+00, -1.562048246801148e+00, -2.941130920436040e-01, -2.938395484703635e-01, -2.075003685522391e-01, -3.255390102012615e-02, -7.032069771007012e-02, -1.032459601256456e-03, -3.423264030528618e-02, -3.398366356709698e-02, -6.896237038913130e-04, -4.902594469249724e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_c09x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_c09x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.148205945225265e-04, 0.000000000000000e+00, -1.144101253199179e-04, -4.883254913478470e-04, 0.000000000000000e+00, -4.866798846118410e-04, -1.087319378412462e-01, 0.000000000000000e+00, -1.087805731398783e-01, -1.724447970111858e+00, 0.000000000000000e+00, -3.103958235626683e-27, -6.022506160555169e+01, 0.000000000000000e+00, 0.000000000000000e+00, -3.878561986130390e-24, 0.000000000000000e+00, -1.630672568433713e-25, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
