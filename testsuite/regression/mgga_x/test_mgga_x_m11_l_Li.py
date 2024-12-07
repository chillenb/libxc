
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_m11_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.344061232280401e+00, -1.552675380014093e+00, -4.493977361807757e-01, -1.782837910074042e-01, -1.437531747988534e-01, -8.362746908957941e-02, -1.563313885684010e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_m11_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.467432303304916e+00, -1.457670325012161e+00, -2.540370876898978e+00, -2.542697995005335e+00, -4.255132496365466e-01, -4.353027476332493e-01, -3.921387784722619e-01, -1.056359385056528e-01, -1.091828854675515e-01, -3.378753941006301e-03, -1.113663747784837e-01, -1.102296837216500e-01, -2.256860022920554e-03, -1.604422515565548e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.753674973427964e-03, 0.000000000000000e+00, -2.756680451005103e-03, -1.635683643402403e-03, 0.000000000000000e+00, -1.641485609714367e-03, -2.299447737752548e-02, 0.000000000000000e+00, -2.411468641316891e-02, 1.145773276757482e+02, 0.000000000000000e+00, -6.406418716089484e+00, -2.194530619176834e+02, 0.000000000000000e+00, -4.085260973804800e+01, -6.515504838483562e+00, 0.000000000000000e+00, -6.081205980762871e+00, -2.973916670973419e+01, 0.000000000000000e+00, -4.256851317378591e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.425445892588342e-02, -1.494274323118771e-02, 1.015668900417028e-01, 1.017792744206984e-01, -1.361357409340518e-02, -1.136161046501521e-02, -2.206561349835494e+00, -3.754835591890785e-05, 2.015564977278275e-01, -7.611298876859479e-09, -1.869732624433912e-08, -4.056558628383056e-05, -1.036689211897359e-19, -8.491552422247568e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
