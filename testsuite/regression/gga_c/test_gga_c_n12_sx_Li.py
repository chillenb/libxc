
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_n12_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.221613486277920e-01, -8.072964913021405e-02, 6.365691594341875e-02, -3.667774372269773e-02, 1.021438236135225e-02, -2.662106244479953e-03, 2.130194179689505e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_n12_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.296542848616519e-01, -2.295115831100495e-01, -2.167377360358149e-01, -2.166975543512242e-01, -2.174986305298002e-02, -2.198367601632610e-02, -6.689994913326980e-02, -2.645535185326840e-01, -3.437784729206566e-03, -1.772763020813313e-01, -3.126248045924784e-03, -3.432192209597900e-03, 1.327256622452248e-04, -2.705892483013738e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_n12_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.637194978880821e-04, 0.000000000000000e+00, 1.631541481650682e-04, 6.718756499842168e-04, 0.000000000000000e+00, 6.700232710128123e-04, 4.580507271126568e-02, 0.000000000000000e+00, 4.588333639798933e-02, 1.181958521669209e+01, 0.000000000000000e+00, -1.974707715775446e+02, 3.425026554679605e+01, 0.000000000000000e+00, -2.346054208161692e+04, -2.367139508945870e+00, 0.000000000000000e+00, -2.514640095588641e+00, -3.735880552050893e+00, 0.000000000000000e+00, -6.162800956343573e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
