
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.277283509714928e-01, -5.197604661235036e-01, -2.611253294981679e-01, -5.930213107752066e-02, -4.279668204464832e-02, -3.435107782792134e-02, -6.431615983047756e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.275966204055354e-01, -8.307521837383839e-01, -4.185128000562382e-01, -4.177249042065978e-01, -2.212273235594804e-01, -2.208613071484679e-01, -7.975254123776399e-02, -4.337481078483210e-02, -1.226736638953754e-02, -1.390050765898961e-03, -4.599587653555078e-02, -4.525065935037917e-02, -9.284971861691051e-04, -6.600737013261233e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.925464630249722e-05, 0.000000000000000e+00, -8.895598114672394e-05, -4.092824518593609e-04, 0.000000000000000e+00, -4.077376270937791e-04, -4.690585993014409e-02, 0.000000000000000e+00, -4.671033642634010e-02, -1.463640279289956e+00, 0.000000000000000e+00, -4.635092125112157e-01, -3.598852317870935e+01, 0.000000000000000e+00, -2.976447745130457e+00, -4.728210023410606e-01, 0.000000000000000e+00, -4.397478109238108e-01, -2.166765690992259e+00, 0.000000000000000e+00, -3.101495088372151e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_2x_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_2x_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.129530808991443e-03, 6.464505325806229e-03, -2.876140270745247e-02, -2.892946211309138e-02, -5.544046337545132e-03, -5.694419553069973e-03, 1.046724733429316e-01, -4.168412036591906e-05, -2.373610999733395e-01, -8.534771703338178e-09, -2.070846164875361e-08, -4.499159496359889e-05, -1.158474739490005e-19, -9.521882508407342e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
